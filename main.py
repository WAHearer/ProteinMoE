import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
from esm import pretrained
from transformers import AutoModelForMaskedLM, AutoTokenizer
from foldseek_util import get_struc_seq
from Bio import SeqIO
import re
import random
import math

alphabet = "LAGVSERTIDPKQNFYMHWC"
foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"
num_experts = 4
epochs = 30 #训练轮数
d_model = 20 * num_experts #encoder维度
nhead = 8 #多头注意力头数
dim_feedforward = 512 #前馈网络维度
dropout = 0.3 #dropout率
num_layers = 6 #Transformer encoder层数
device = torch.device("cuda:1")
temp = 1
eps = 1e-8

#ESM
model_ESM, ESM_alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
ESM_batch_converter = ESM_alphabet.get_batch_converter()
model_ESM = model_ESM.to(device)
for param in model_ESM.parameters():
    param.requires_grad = False
model_ESM.eval()

#SaProt
model_SaProt = AutoModelForMaskedLM.from_pretrained("saprot").to(device)
SaProt_tokenizer = AutoTokenizer.from_pretrained("saprot")
for param in model_SaProt.parameters():
    param.requires_grad = False
model_SaProt.eval()

#ProSST-2048
model_ProSST_2048 = AutoModelForMaskedLM.from_pretrained("ProSST-2048", trust_remote_code=True).to(device)
ProSST_tokenizer = AutoTokenizer.from_pretrained("ProSST-2048", trust_remote_code=True)
for param in model_ProSST_2048.parameters():
    param.requires_grad = False
model_ProSST_2048.eval()

#ProSST-4096
model_ProSST_4096 = AutoModelForMaskedLM.from_pretrained("ProSST-4096", trust_remote_code=True).to(device)
for param in model_ProSST_4096.parameters():
    param.requires_grad = False
model_ProSST_4096.eval()

def tokenize_structure_sequence(structure_sequence):
    shift_structure_sequence = [i + 3 for i in structure_sequence]
    shift_structure_sequence = [1, *shift_structure_sequence, 2]
    return torch.tensor(
        [
            shift_structure_sequence,
        ],
        dtype=torch.long,
    )
esm_vocab_to_prosst = [12, 3, 8, 20, 18, 6, 17, 19, 10, 5, 15, 11, 16, 14, 7, 22, 13, 9, 21, 4]

def calc(row, sequence, token_probs, alphabet, offset_idx):
    score=0
    for mutation in row.split(":"):
        wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

        wt_encoded, mt_encoded = alphabet.get_idx(wt)-4, alphabet.get_idx(mt)-4
        # add 1 for BOS
        score += torch.log(token_probs[0, 1 + idx, mt_encoded] / token_probs[0, 1 + idx, wt_encoded])
    return score.item()

def get_mutated_sequence(focus_seq, mutant, start_idx=1, AA_vocab="ACDEFGHIKLMNPQRSTVWY"):
    mutated_seq = list(focus_seq)
    for mutation in mutant.split(":"):
        try:
            from_AA, position, to_AA = mutation[0], int(mutation[1:-1]), mutation[-1]
        except:
            print("Issue with mutant: " + str(mutation))
        relative_position = position - start_idx
        assert (from_AA == focus_seq[relative_position]), "Invalid from_AA or mutant position: " + str(
            mutation) + " from_AA: " + str(from_AA) + " relative pos: " + str(relative_position) + " focus_seq: " + str(
            focus_seq)
        assert (to_AA in AA_vocab), "Mutant to_AA is invalid: " + str(mutation)
        mutated_seq[relative_position] = to_AA
    return "".join(mutated_seq)
    
class Router(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.Linear(d_model * 2, num_experts)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    
class MoE(nn.Module):
    def __init__(self, num_experts=2):
        super().__init__()
        self.num_experts = num_experts
        self.router = Router(num_experts)

    def forward(self, ESM_input, SaProt_input, ProSST_input_ids, ProSST_attention_mask, ProSST_2048_ss_input_ids, ProSST_4096_ss_input_ids, start_pos=0):
        #ESM
        ESM_output = model_ESM(ESM_input)["logits"][:, :, 4: 24]
        ESM_output = F.softmax(ESM_output, dim=-1)

        #SaProt
        SaProt_output = model_SaProt(**SaProt_input).logits
        SaProt_output = SaProt_output.softmax(dim=-1)
        SaProt_output_aligned = torch.zeros(SaProt_output.shape[0], SaProt_output.shape[1], 20).to(device)
        for i in range(20):
            st = SaProt_tokenizer.get_vocab()[alphabet[i] + foldseek_struc_vocab[0]]
            SaProt_output_aligned[:, :, i] = SaProt_output[:, :,st: st + len(foldseek_struc_vocab)].sum(dim=-1)
        if start_pos > 0:
            zeros = torch.zeros(SaProt_output_aligned.shape[0], start_pos, 20).to(device)
            SaProt_output_aligned = torch.cat([zeros, SaProt_output_aligned], dim=1)
        if SaProt_output_aligned.shape[1] < ESM_output.shape[1]:
            zeros = torch.zeros(SaProt_output_aligned.shape[0], ESM_output.shape[1] - SaProt_output_aligned.shape[1], 20).to(device)
            SaProt_output_aligned = torch.cat([SaProt_output_aligned, zeros], dim=1)

        #ProSST-2048
        ProSST_2048_output = model_ProSST_2048(
            input_ids = ProSST_input_ids,
            attention_mask = ProSST_attention_mask,
            ss_input_ids = ProSST_2048_ss_input_ids,
            labels = ProSST_input_ids
        ).logits
        ProSST_2048_output = ProSST_2048_output.softmax(dim=-1)
        ProSST_2048_output_aligned = torch.zeros(ProSST_2048_output.shape[0], ProSST_2048_output.shape[1], 20).to(device)
        for i in range(20):
            ProSST_2048_output_aligned[:, :, i] = ProSST_2048_output[:, :, esm_vocab_to_prosst[i]]
        if start_pos > 0:
            zeros = torch.zeros(ProSST_2048_output_aligned.shape[0], start_pos, 20).to(device)
            ProSST_2048_output_aligned = torch.cat([zeros, ProSST_2048_output_aligned], dim=1)
        if ProSST_2048_output_aligned.shape[1] < ESM_output.shape[1]:
            zeros = torch.zeros(ProSST_2048_output_aligned.shape[0], ESM_output.shape[1] - ProSST_2048_output_aligned.shape[1], 20).to(device)
            ProSST_2048_output_aligned = torch.cat([ProSST_2048_output_aligned, zeros], dim=1)

        #ProSST-4096
        ProSST_4096_output = model_ProSST_4096(
            input_ids = ProSST_input_ids,
            attention_mask = ProSST_attention_mask,
            ss_input_ids = ProSST_4096_ss_input_ids,
            labels = ProSST_input_ids
        ).logits
        ProSST_4096_output = ProSST_4096_output.softmax(dim=-1)
        ProSST_4096_output_aligned = torch.zeros(ProSST_4096_output.shape[0], ProSST_4096_output.shape[1], 20).to(device)
        for i in range(20):
            ProSST_4096_output_aligned[:, :, i] = ProSST_4096_output[:, :, esm_vocab_to_prosst[i]]
        if start_pos > 0:
            zeros = torch.zeros(ProSST_4096_output_aligned.shape[0], start_pos, 20).to(device)
            ProSST_4096_output_aligned = torch.cat([zeros, ProSST_4096_output_aligned], dim=1)
        if ProSST_4096_output_aligned.shape[1] < ESM_output.shape[1]:
            zeros = torch.zeros(ProSST_4096_output_aligned.shape[0], ESM_output.shape[1] - ProSST_4096_output_aligned.shape[1], 20).to(device)
            ProSST_4096_output_aligned = torch.cat([ProSST_4096_output_aligned, zeros], dim=1)

        outputs = torch.cat([ESM_output, SaProt_output_aligned, ProSST_2048_output_aligned, ProSST_4096_output_aligned], dim=-1)
        router_output = self.router(outputs)
        weights = F.softmax(router_output/temp, dim=-1)
        stacked_outputs = torch.cat([ESM_output, SaProt_output_aligned, ProSST_2048_output_aligned, ProSST_4096_output_aligned], dim=0)
        weights = weights.squeeze(0)
        weighted_sum = torch.einsum('ijk,ji->jk', stacked_outputs, weights)
        weighted_sum = weighted_sum.unsqueeze(0)
        return weighted_sum, ESM_output, SaProt_output_aligned, ProSST_2048_output_aligned, ProSST_4096_output_aligned, weights

def main():
    model = MoE(num_experts=num_experts)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.NLLLoss()
    st, ed = [], []
    seqs = []
    with open("cath_data/cath-domain-seqs-v4_4_0.fa", "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            id = re.findall(r"(?<=4_4_0\|).*(?=/)", record.id)[0]
            if not os.path.exists(f"cath_data/dompdb/{id}"):
                continue
            struc_range = re.findall(r"(?<=/).*", record.id)[0]
            struc_range = re.sub(r'[^0-9-]', '', struc_range)
            st.append(int(struc_range.split("-")[0]))
            ed.append(min(int(struc_range.split("-")[1]), len(record.seq)))
            seqs.append((id, str(record.seq)))
    f = open("process","w")
    for epoch in range(epochs):
        cnt, sumloss, index = 0, 0, 0
        for id, record in seqs:
            if id[4:5].isdigit():
                index += 1
                continue
            if not os.path.exists("cath_data/struc_seq_2048/"+id) or not os.path.exists("cath_data/struc_seq_4096/"+id):
                index += 1
                continue
            
            #get masked sequence
            seq = record
            pick_pos = random.sample(range(st[index] - 1,ed[index]), round(0.15 * (ed[index] - st[index] + 1)))
            mask_pos = random.sample(pick_pos, round(0.8 * len(pick_pos)))
            left_pos = [pos for pos in pick_pos if pos not in mask_pos]
            random_pos = random.sample(left_pos, round(0.5 * len(left_pos)))
            for pos in mask_pos:
                seq = seq[:pos] + "#" + seq[pos+1:]
            for pos in random_pos:
                seq = seq[:pos] + random.choice(alphabet) + seq[pos+1:]

            #ESM
            ESM_seq = seq.replace("#", "<mask>")
            data = [
                ("protein1", ESM_seq),
            ]
            _, _, ESM_input = ESM_batch_converter(data)
            data = [
                ("protein1", record),
            ]
            _, _, truth = ESM_batch_converter(data)
            ESM_input = ESM_input.to(device)
            truth = truth.to(device)

            #SaProt
            struc_seq = get_struc_seq("foldseek/bin/foldseek", f"cath_data/dompdb/{id}", [id[4:5]], plddt_mask=False, plddt_threshold=70)[id[4:5]][1].lower()
            seq_split = seq[st[index] - 1: ed[index]]
            SaProt_seq = "".join([a + b for a, b in zip(seq_split, struc_seq)])
            SaProt_tokens = SaProt_tokenizer.tokenize(SaProt_seq)
            SaProt_mask_seq = " ".join(SaProt_tokens)
            SaProt_input = SaProt_tokenizer(SaProt_mask_seq, return_tensors="pt")
            SaProt_input = {k: v.to(device) for k, v in SaProt_input.items()}

            #ProSST-2048
            ProSST_seq = seq_split.replace("#", "<mask>")
            ProSST_tokenized_results = ProSST_tokenizer([ProSST_seq], return_tensors="pt")
            ProSST_input_ids = ProSST_tokenized_results["input_ids"].to(device)
            ProSST_attention_mask = ProSST_tokenized_results["attention_mask"].to(device)
            with open("cath_data/struc_seq_2048/"+id, "r") as file:
                struc_seq_2048 = file.readline()
            struc_seq_2048 = [int(i) for i in struc_seq_2048.split()]
            if len(struc_seq_2048) > len(seq_split):
                struc_seq_2048 = struc_seq_2048[:len(seq_split)]
            if len(struc_seq_2048) < len(seq_split):
                index += 1
                continue
            ProSST_2048_ss_input_ids = tokenize_structure_sequence(struc_seq_2048).to(device)

            #ProSST-4096
            with open("cath_data/struc_seq_4096/"+id, "r") as file:
                struc_seq_4096 = file.readline()
            struc_seq_4096 = [int(i) for i in struc_seq_4096.split()]
            if len(struc_seq_4096) > len(seq_split):
                struc_seq_4096 = struc_seq_4096[:len(seq_split)]
            if len(struc_seq_4096) < len(seq_split):
                index += 1
                continue
            ProSST_4096_ss_input_ids = tokenize_structure_sequence(struc_seq_4096).to(device)
            
            #train
            output, esm, saprot, prosst_2048, prosst_4096, weights = model.forward(ESM_input, SaProt_input, ProSST_input_ids, ProSST_attention_mask, ProSST_2048_ss_input_ids, ProSST_4096_ss_input_ids, start_pos=st[index] - 1)
            #for pos in pick_pos:
            #    print(f"{esm[0, pos, truth[0, pos] - 4]} {saprot[0, pos, truth[0, pos] - 4]} {prosst_2048[0, pos, truth[0, pos] - 4]} {prosst_4096[0, pos, truth[0, pos] - 4]} {output[0, pos, truth[0, pos] - 4]}")
            #    print(weights[pos])
            pick_pos = [p + 1 for p in pick_pos]
            preds=torch.zeros(len(pick_pos),20).to(device)
            for i in range(len(pick_pos)):
                preds[i]=output[0, pick_pos[i]]
            labels = torch.tensor([truth[0, p] - 4 for p in pick_pos]).to(device)
            preds = torch.log(preds + eps)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cnt += 1
            sumloss += loss
            
            if cnt % 1000 == 0:
                f.write(f"Epoch {epoch+1} Processed {index+1}/{len(seqs)} sequences, avg Loss: {sumloss/cnt:.4f}\n")
                f.flush()
            index += 1
        torch.save(model.state_dict(), f"train/model_epoch_{epoch+1}.pth")
    f.close()


    model.eval()
    mapping_protein_seq_DMS = pd.read_csv("DMS_substitutions.csv")
    list_DMS = mapping_protein_seq_DMS["DMS_id"]
    f=open("MoE_result","a")
    sum=0
    for DMS_index in range(217):
        DMS_id = list_DMS[DMS_index]
        DMS_file_name = mapping_protein_seq_DMS["DMS_filename"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0]
        pdb_file_name = mapping_protein_seq_DMS["pdb_file"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0]
        prosst_struc_file_name = pdb_file_name.replace(".pdb", "")
        seq = mapping_protein_seq_DMS["target_seq"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0].upper()
        df = pd.read_csv("DMS_ProteinGym_substitutions" + os.sep + DMS_file_name, low_memory=False)
        data = [
            ("protein1", seq),
        ]
        _, _, ESM_input = ESM_batch_converter(data)
        ESM_input = ESM_input.to(device)
        df['mutated_sequence'] = df['mutant'].apply(
            lambda x: get_mutated_sequence(seq, x))
        pdb_range = mapping_protein_seq_DMS["pdb_range"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0]
        pdb_range = [int(x) for x in pdb_range.split("-")]
        target_seq_split = seq[pdb_range[0] - 1:pdb_range[1]]
        struc_seq = get_struc_seq("foldseek/bin/foldseek", "ProteinGym_AF2_structures/" + pdb_file_name, ["A"], plddt_mask=True, plddt_threshold=70)["A"][1].lower()
        SaProt_seq = "".join([a + b for a, b in zip(target_seq_split, struc_seq)])
        SaProt_tokens = SaProt_tokenizer.tokenize(SaProt_seq)
        with open("ProteinGym_struc_2048/"+prosst_struc_file_name) as file:
            struc_seq_2048 = file.readline()
        struc_seq_2048 = [int(i) for i in struc_seq_2048.split()]
        ProSST_2048_ss_input_ids = tokenize_structure_sequence(struc_seq_2048).to(device)
        with open("ProteinGym_struc_4096/"+prosst_struc_file_name) as file:
            struc_seq_4096 = file.readline()
        struc_seq_4096 = [int(i) for i in struc_seq_4096.split()]
        ProSST_4096_ss_input_ids = tokenize_structure_sequence(struc_seq_4096).to(device)
        model_scores = []
        for mut_info in df["mutant"]:
            SaProt_tokens_masked = SaProt_tokens.copy()
            ESM_input_masked = ESM_input.clone()
            ProSST_seq = str(target_seq_split)
            for single in mut_info.split(":"):
                ESM_input_masked[0, int(single[1:-1])] = ESM_alphabet.mask_idx
                pos = int(single[1:-1]) - pdb_range[0] + 1
                SaProt_tokens_masked[pos - 1] = "#" + SaProt_tokens_masked[pos - 1][-1]
                ProSST_seq = ProSST_seq[:pos-1] + "#" + ProSST_seq[pos:]
            mask_seq = " ".join(SaProt_tokens_masked)
            SaProt_inputs = SaProt_tokenizer(mask_seq, return_tensors="pt")
            SaProt_inputs = {k: v.to(device) for k, v in SaProt_inputs.items()}
            ProSST_seq = ProSST_seq.replace("#", "<mask>")
            ProSST_tokenized_results = ProSST_tokenizer([ProSST_seq], return_tensors="pt")
            ProSST_input_ids = ProSST_tokenized_results["input_ids"].to(device)
            ProSST_attention_mask = ProSST_tokenized_results["attention_mask"].to(device)

            output, esm, saprot, prosst_2048, prosst_4096, weights = model.forward(ESM_input_masked, SaProt_inputs, ProSST_input_ids, ProSST_attention_mask, ProSST_2048_ss_input_ids, ProSST_4096_ss_input_ids, start_pos=pdb_range[0] - 1)
            model_scores.append(calc(mut_info, seq, output, ESM_alphabet, 1))
        df['MoE_score'] = model_scores
        corr, p_value = spearmanr(df['MoE_score'], df['DMS_score'])
        scoring_filename = "MoE_output" + os.sep + DMS_id + '.csv'
        df[['mutant', 'DMS_score', 'MoE_score']].to_csv(scoring_filename, index=False)
        f.write(f"样本 {DMS_id} 的Spearman相关系数：{corr:.3f}\n")
        f.flush()
        sum+=corr
    f.write(f"平均Spearman相关系数：{sum/217:.3f}\n")
    f.flush()
    f.close()
if __name__ == "__main__":
    main()
