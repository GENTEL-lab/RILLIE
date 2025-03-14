import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from Bio import SeqIO
import sys

modelgenerator_path = "./RILLIE/model/LLModel"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print(f"Added {project_root} to PYTHONPATH")
if modelgenerator_path not in sys.path:
    sys.path.insert(0, modelgenerator_path)
    
# ----- RhoDesign model related imports -----
from RhoDesign import RhoDesignModel
from alphabet import Alphabet
from util import load_structure, extract_coords_from_structure, CoordBatchConverter

# ----- LLM (AIDO) model related imports -----
from LLModel.modelgenerator.tasks import MLM
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

# Set random seed
random.seed(0)

# Global variables: Alphabet and batch_converter for RhoDesign model
alphabet = Alphabet(['A', 'G', 'C', 'U', 'X'])
batch_converter = CoordBatchConverter(alphabet)

# Parameter class for RhoDesign model
class args_class:  
    def __init__(self, encoder_embed_dim, decoder_embed_dim, dropout):
        self.local_rank = int(os.getenv("LOCAL_RANK", -1))
        self.device_id = [0]  # simplified version
        self.epochs = 100
        self.lr = 1e-5
        self.batch_size = 1
        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.dropout = dropout
        self.gvp_top_k_neighbors = 15
        self.gvp_node_hidden_dim_vector = 256
        self.gvp_node_hidden_dim_scalar = 512
        self.gvp_edge_hidden_dim_scalar = 32
        self.gvp_edge_hidden_dim_vector = 1
        self.gvp_num_encoder_layers = 3
        self.gvp_dropout = 0.1
        self.encoder_layers = 3
        self.encoder_attention_heads = 4
        self.attention_dropout = 0.1
        self.encoder_ffn_embed_dim = 512
        self.decoder_layers = 3
        self.decoder_attention_heads = 4
        self.decoder_ffn_embed_dim = 512

# ---------- RhoDesign model scoring functions ----------

def get_sequence_loss(model, batch, device):
    # Process the batch using the global batch_converter
    coords, confidence, strs, tokens, padding_mask, ss_ct_map = batch_converter(batch, device=device)
    c = coords[:, :, [0, 1, 2], :]  # four backbone atoms
    adc = coords[:, :, :, :]
    padding_mask = padding_mask.bool()
    prev_output_tokens = tokens[:, :-1].to(device)
    target = tokens[:, 1:]
    target_padding_mask = (target == alphabet.padding_idx)
    logits, _ = model.forward(c, adc, ss_ct_map, padding_mask, confidence, prev_output_tokens)
    loss = F.cross_entropy(logits, target, reduction='none')
    loss = loss[0].cpu().detach().numpy()
    target_padding_mask = target_padding_mask[0].cpu().numpy()
    return loss, target_padding_mask

def score_sequence(model, batch, device):
    loss, target_padding_mask = get_sequence_loss(model, batch, device)
    # Compute the full sequence log-likelihood (negative average loss)
    ll_fullseq = -np.sum(loss * ~target_padding_mask) / np.sum(~target_padding_mask)
    return ll_fullseq

def score_backbone(model, coords, seq, ss_ct_map, device):
    batch = [(coords, None, seq, ss_ct_map)]
    ll = score_sequence(model, batch, device)
    ppl = np.exp(-ll)  # perplexity
    return ppl, ll

def init_rhodesign_model(model_path, device, embed_dim=512, dropout=0.1):
    # Initialize the model and load pre-trained weights
    args = args_class(embed_dim, embed_dim, dropout)
    dictionary = Alphabet(['A', 'G', 'C', 'U', 'X'])
    model = RhoDesignModel(args, dictionary).to(device)
    temp = torch.load(model_path)
    model.load_state_dict(temp)
    model.eval()
    return model

def eval_ppl_for_sequence_custom(model, pdb_list, seq, device, pdb_dir, ss_dir):
    """
    For a single sequence, iterate through pdb_list (each pdb file corresponds to a structure),
    compute the score for the sequence on each structure, and return the average perplexity and log-likelihood.
    """
    ppl_scores = []
    ll_scores = []
    for pdb_id in tqdm(pdb_list, desc="Processing pdb files for sequence"):
        fpath = os.path.join(pdb_dir, pdb_id + '.pdb')
        ss_path = os.path.join(ss_dir, pdb_id + '.npy')
        s = load_structure(fpath)
        coords, _ = extract_coords_from_structure(s)
        ss_ct_map = np.load(ss_path)
        ppl_v, ll_v = score_backbone(model, coords, seq, ss_ct_map, device)
        ppl_scores.append(ppl_v)
        ll_scores.append(ll_v)
    return np.mean(ppl_scores), np.mean(ll_scores)

# ---------- LLM (AIDO) model scoring functions ----------

def init_llm_model(llm_name="aido_rna_1b600m", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLM.from_config({"model.backbone": llm_name}).eval().to(device)
    return model

def score_llm_model_for_sequence(model, seq, device):
    # Replace 'U' with 'T' as required by the model
    seq_mod = seq.replace('U', 'T')
    collated_batch = model.transform({"sequences": [seq_mod]})
    input_ids = collated_batch["input_ids"].to(device)
    attention_mask = collated_batch["attention_mask"].to(device)
    logits = model(collated_batch)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    logits_flat = logits.view(-1, logits.size(-1))
    labels_flat = input_ids.view(-1)
    loss_all = loss_fn(logits_flat, labels_flat)
    mask_flat = attention_mask.view(-1)
    loss_masked = loss_all[mask_flat == 1]
    loss = loss_masked.mean()
    loglikelihood = -loss
    return loglikelihood.item()

# ---------- Main function: Combine scoring from two models and filter sequences in top x% for both models ----------

def main(input_fasta, output_fasta, top_percent, rhodesign_model_path, pdb_dir, ss_dir, llm_name="aido_rna_1b600m"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Initialize the two models
    print("Initializing IFM model...")
    rhodesign_model = init_rhodesign_model(rhodesign_model_path, device)
    print("Initializing LLM model...")
    llm_model = init_llm_model(llm_name, device)
    
    # Get all pdb filenames (without extension) from the pdb directory
    pdb_list = [os.path.splitext(f)[0] for f in os.listdir(pdb_dir) if f.endswith('.pdb')]
    
    # Read all sequences from the input FASTA file
    records = list(SeqIO.parse(input_fasta, "fasta"))
    print(f"Found {len(records)} sequences in the input fasta.")
    
    scores_rho = []   # Store RhoDesign scores (log-likelihood, higher is better)
    scores_llm = []   # Store LLM scores (log-likelihood)
    
    # Score each sequence with both models
    for record in records:
        seq = str(record.seq)
        print(f"Scoring sequence {record.id}...")
        # RhoDesign scoring (average over all pdb structures)
        _, rho_ll = eval_ppl_for_sequence_custom(rhodesign_model, pdb_list, seq, device, pdb_dir, ss_dir)
        scores_rho.append(rho_ll)
        # LLM scoring
        llm_score = score_llm_model_for_sequence(llm_model, seq, device)
        scores_llm.append(llm_score)
        print(f"Sequence {record.id}: IFM score = {rho_ll}, LLM score = {llm_score}")
    
    # Compute thresholds: take the lowest score within the top_percent for each model
    top_p = float(top_percent)
    threshold_rho = np.percentile(scores_rho, 100 - top_p)
    threshold_llm = np.percentile(scores_llm, 100 - top_p)
    print(f"Threshold for IFM (top {top_p}%): {threshold_rho}")
    print(f"Threshold for LLM (top {top_p}%): {threshold_llm}")
    
    # Filter sequences that meet both thresholds
    selected_records = []
    for record, score_r, score_ll in zip(records, scores_rho, scores_llm):
        if score_r >= threshold_rho and score_ll >= threshold_llm:
            selected_records.append(record)
    
    print(f"Selected {len(selected_records)} sequences out of {len(records)}.")
    
    # Write the filtered sequences to the output FASTA file
    with open(output_fasta, "w") as f:
        SeqIO.write(selected_records, f, "fasta")
    
    print(f"Output written to {output_fasta}.")

if __name__ == "__main__":
    # Example usage, modify paths and parameters as needed
    input_fasta = "./RILLIE/data/example_pepper.fasta"                  # Input FASTA file path
    output_fasta = "./RILLIE/data/output.fasta"    # Output filtered FASTA file path
    top_percent = 40                             # Select top 10% sequences based on scores
    rhodesign_model_path = "./RILLIE/model/IFModel/checkpoint/ss_apexp_best.pth"  # RhoDesign model checkpoint path
    pdb_dir = "./RILLIE/model/IFModel/data/test"     # Directory containing pdb files
    ss_dir = "./RILLIE/model/IFModel/data/test_ss"     # Directory containing corresponding .npy files
    llm_name = "aido_rna_1b600m"                   # LLM model name
    
    main(input_fasta, output_fasta, top_percent, rhodesign_model_path, pdb_dir, ss_dir, llm_name)
