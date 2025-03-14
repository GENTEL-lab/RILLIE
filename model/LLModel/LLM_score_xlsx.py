import json
from modelgenerator.tasks import MLM
from Bio import SeqIO  # 用于读取FASTA文件
import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import pandas as pd  # 用于生成Excel文件

def calculate_loglikelihood(fasta_file, xlsx_file, model_name="aido_rna_1b600m"):
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型并移动到GPU
    model = MLM.from_config({"model.backbone": model_name}).eval().to(device)
    
    # 读取FASTA文件中的序列
    sequences = []
    names = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        names.append(record.id)
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

    # 创建一个DataFrame来保存数据
    df = pd.DataFrame({
        'Name': names,
        'Sequence': sequences,
        'Log-Likelihood': loglikelihoods
    })

    # 保存为xlsx文件
    df.to_excel(xlsx_file, index=False)

    # 输出文件路径
    print(f"Results saved to {xlsx_file}")

# 示例调用
if __name__ == "__main__":
    fasta_file = "/home/zaitpub04/hyj/RhoDesign/RhoDesign/src/choose_mutation/WETTEST.txt/broccoliR1.fasta"  # 输入的FASTA文件路径
    xlsx_file = "/home/zaitpub04/hyj/ModelGenerator/broccoliR1_AIDO.xlsx"  # 输出的Excel文件路径
    calculate_loglikelihood(fasta_file, xlsx_file)
