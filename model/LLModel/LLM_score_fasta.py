import json
from modelgenerator.tasks import SequenceRegression, MLM
from Bio import SeqIO  # 用于读取FASTA文件
import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

def calculate_loglikelihood(fasta_file, json_file, model_name="aido_rna_1b600m"):
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型并移动到GPU
    model = MLM.from_config({"model.backbone": model_name}).eval().to(device)
    
    # 读取FASTA文件中的序列
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append(str(record.seq))
    
    loglikelihoods = []
    
    # 使用模型的collate方法处理序列
    for seq in sequences:
        collated_batch = model.transform({"sequences": [seq.replace('U', 'T')]})
        
        # 将输入数据移动到GPU
        input_ids = collated_batch["input_ids"].to(device)
        attention_mask = collated_batch["attention_mask"].to(device)
        
        logits = model(collated_batch)
        
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

        logits_flat = logits.view(-1, logits.size(-1))  
        labels_flat = input_ids.view(-1)  

        loss_all = loss_fn(logits_flat, labels_flat)  

        # 展开 attention_mask 并选择非 padding 的部分
        mask_flat = attention_mask.view(-1)  # (batch_size * seq_len)
        loss_masked = loss_all[mask_flat == 1]  # 仅保留非 padding 部分的 loss

        loss = loss_masked.mean()
        loglikelihood = -loss

        print("Log-likelihood:", loglikelihood.item())
        loglikelihoods.append(loglikelihood.item())

    if os.path.exists(json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
    else:
        data = {}

    key = 'AIDO_score'
    # 添加新的键值对
    data[key] = loglikelihoods

    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

    # 示例调用
if __name__ == "__main__":
    fasta_file = "/home/zaitpub04/hyj/RhoDesign/benchmark/fasta/myo.fasta"  # 输入的FASTA文件路径
    json_file = "/home/zaitpub04/hyj/RhoDesign/benchmark/json/myo.json"      # 输出的JSON文件路径
    calculate_loglikelihood(fasta_file, json_file)
    