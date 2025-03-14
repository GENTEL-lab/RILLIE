from RhoDesign import RhoDesignModel
from alphabet import Alphabet
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
from util import load_structure, extract_coords_from_structure, seq_rec_rate,CoordBatchConverter
import os
from scipy.stats import spearmanr
random.seed(0)
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors
import scipy.stats as stats
from matplotlib import rcParams
import pandas as pd
_device = 0
alphabet = Alphabet(['A','G','C','U','X'])
batch_converter = CoordBatchConverter(alphabet)


class args_class:  
    def __init__(self, encoder_embed_dim, decoder_embed_dim, dropout):
        self.local_rank = int(os.getenv("LOCAL_RANK", -1))
        self.device_id = [0, 1, 2, 3, 4, 5, 6, 7]
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

def get_sequence_loss(model, batch , _device):
    device = _device
    # batch_converter = CoordBatchConverter(alphabet)
    
    coords, confidence, strs, tokens, padding_mask,ss_ct_map = batch_converter(
        batch, device=device)
    
    c = coords[:,:,[0,1,2],:] # the four backbone atoms
    adc = coords[:,:,:,:] # eight atoms which are used to compute dihedral angles
    padding_mask = padding_mask.bool()

    prev_output_tokens = tokens[:, :-1].to(device)
    target = tokens[:, 1:]
    target_padding_mask = (target == alphabet.padding_idx)
    logits, _ = model.forward(c, adc,ss_ct_map,padding_mask, confidence, prev_output_tokens)
    loss = F.cross_entropy(logits, target, reduction='none')
    loss = loss[0].cpu().detach().numpy()
    target_padding_mask = target_padding_mask[0].cpu().numpy()
    return loss, target_padding_mask

def score_sequence(model, batch,_device):
    loss, target_padding_mask = get_sequence_loss(model, batch,_device)
    ll_fullseq = -np.sum(loss * ~target_padding_mask) / np.sum(~target_padding_mask)
    return ll_fullseq

def score_backbone(model, coords, seq, ss_ct_map, _device):
    batch = [(coords, None, seq,ss_ct_map)]
    ll= score_sequence(model, batch,_device) 
    ppl = np.exp(-ll)
    return ppl , ll

def eval_ppl(model, pdb_list, model_path, mut):
    temp = torch.load(model_path)
    model.load_state_dict(temp)
    model.eval()

    with torch.no_grad():
        ## change the path to your own path
        pfile = './RILLIE/model/IFM/data/test/'
        ssfile = './RILLIE/model/IFM/data/test_ss/'
        ppl = []
        ll=[]
        for i in tqdm(pdb_list):
            fpath = pfile + i + '.pdb'
            ss_path = ssfile + i + '.npy'
            s = load_structure(fpath)
            coords, seq = extract_coords_from_structure(s)
            ss_ct_map = np.load(ss_path)
            ppl_v ,ll_v= score_backbone(model, coords, mut, ss_ct_map, _device)
            ppl.append(ppl_v)
            ll.append(ll_v)
    return np.mean(ppl) ,np.mean(ll)

## change the path to your own path of fasta file
fasta_path = './RILLIE/data/broccoli.fasta'
with open(fasta_path, 'r') as f:
    fasta_content = f.read()

fasta_sequences = fasta_content.split('>')[1:]
names = []
sequences = []
for seq in fasta_sequences:
    print(seq)
    name, seq = seq.split('\n', 1)
    names.append(name.strip())
    sequences.append(seq.strip())

args = args_class(512, 512, 0.1)
dictionary = Alphabet(['A', 'G', 'C', 'U', 'X'])
model = RhoDesignModel(args, dictionary).cuda(device=_device)

## change the path to your own path
pdb_list = os.listdir('./RILLIE/model/IFM/data/test/')
pdb_list = [i.split('.')[0] for i in pdb_list]

## change the path to your own path
model_path = './RILLIE/checkpoint/ss_apexp_best.pth'


ll_values = []
for name, seq in zip(names, sequences):
    _, ll = eval_ppl(model, pdb_list, model_path, seq)
    ll_values.append(ll)


df = pd.DataFrame({
    'Name': names,
    'Sequence': sequences,
    'Loglikelihood': ll_values
})
output_xlsx_path = './RILLIE/data/broccoli.xlsx' 
df.to_excel(output_xlsx_path, index=False)

print(f"Results saved to {output_xlsx_path}")