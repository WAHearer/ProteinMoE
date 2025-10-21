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
import warnings
import re
import random
import math
import attr
from esm.models.esm3 import ESM3
from esm.models.vqvae import StructureTokenDecoder, StructureTokenEncoder
from esm.tokenization import get_esm3_model_tokenizers, get_esmc_model_tokenizers
from esm.models.function_decoder import FunctionTokenDecoder
from esm.utils.constants.esm3 import data_root
from esm.utils.constants.models import (
    ESM3_FUNCTION_DECODER_V0,
    ESM3_OPEN_SMALL,
    ESM3_STRUCTURE_DECODER_V0,
    ESM3_STRUCTURE_ENCODER_V0,
    ESMC_300M,
    ESMC_600M,
)
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.utils.structure.protein_chain import ProteinChain

os.environ['TORCH_HOME'] = '/data/zhangtianhe'
alphabet = "LAGVSERTIDPKQNFYMHWC"
foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"
num_experts = 3
epochs = 10 #训练轮数
d_model = 20 * num_experts #encoder维度
nhead = 6 #多头注意力头数
dim_feedforward = 240 #前馈网络维度
dropout = 0.3 #dropout率
num_layers = 6 #Transformer encoder层数
device = torch.device("cuda:2")

#ESM
def ESM3_structure_encoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenEncoder(
            d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
        ).eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_structure_encoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model
def ESM3_structure_decoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = StructureTokenDecoder(d_model=1280, n_heads=20, n_layers=30).eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_structure_decoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model
def ESM3_function_decoder_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = FunctionTokenDecoder().eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_function_decoder_v0.pth",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    return model
def ESM3_sm_open_v0(device: torch.device | str = "cpu"):
    with torch.device(device):
        model = ESM3(
            d_model=1536,
            n_heads=24,
            v_heads=256,
            n_layers=48,
            structure_encoder_fn=ESM3_structure_encoder_v0,
            structure_decoder_fn=ESM3_structure_decoder_v0,
            function_decoder_fn=ESM3_function_decoder_v0,
            tokenizers=get_esm3_model_tokenizers(ESM3_OPEN_SMALL),
        ).eval()
    state_dict = torch.load(
        data_root("esm3") / "data/weights/esm3_sm_open_v1.pth", map_location=device
    )
    model.load_state_dict(state_dict)
    return model

model_ESM = ESM3_sm_open_v0(device)
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

def calc(row, sequence, token_probs, offset_idx):
    score=0
    for mutation in row.split(":"):
        wt, idx, mt = mutation[0], int(mutation[1:-1]) - offset_idx, mutation[-1]
        assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"
        wt_encoded, mt_encoded = alphabet.index(wt), alphabet.index(mt)
        # add 1 for BOS
        score += token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
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

    def forward(self, ESM_input, SaProt_input, ProSST_input_ids, ProSST_attention_mask, ProSST_2048_ss_input_ids, start_pos=0):
        #ESM
        ESM_output = model_ESM.logits(ESM_input, LogitsConfig(sequence=True)).logits.sequence[:, :, 4: 24]
        
        #SaProt
        SaProt_output = model_SaProt(**SaProt_input).logits
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
        ProSST_2048_output_aligned = torch.zeros(ProSST_2048_output.shape[0], ProSST_2048_output.shape[1], 20).to(device)
        for i in range(20):
            ProSST_2048_output_aligned[:, :, i] = ProSST_2048_output[:, :, esm_vocab_to_prosst[i]]
        if start_pos > 0:
            zeros = torch.zeros(ProSST_2048_output_aligned.shape[0], start_pos, 20).to(device)
            ProSST_2048_output_aligned = torch.cat([zeros, ProSST_2048_output_aligned], dim=1)
        if ProSST_2048_output_aligned.shape[1] < ESM_output.shape[1]:
            zeros = torch.zeros(ProSST_2048_output_aligned.shape[0], ESM_output.shape[1] - ProSST_2048_output_aligned.shape[1], 20).to(device)
            ProSST_2048_output_aligned = torch.cat([ProSST_2048_output_aligned, zeros], dim=1)

        outputs = torch.cat([ESM_output, SaProt_output_aligned, ProSST_2048_output_aligned], dim=-1)
        router_output = self.router(outputs)
        stacked_outputs = torch.cat([ESM_output, SaProt_output_aligned, ProSST_2048_output_aligned], dim=0)
        router_output = router_output.squeeze(0)
        weighted_sum = torch.einsum('ijk,ji->jk', stacked_outputs, router_output)
        weighted_sum = weighted_sum.unsqueeze(0)
        return weighted_sum, ESM_output, SaProt_output_aligned, ProSST_2048_output_aligned, router_output

def main():
    model = MoE(num_experts=num_experts)
    model = model.to(device)
    model.load_state_dict(torch.load("train/model_epoch_10.pth"))
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    loss_fn = nn.CrossEntropyLoss()
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
            if not os.path.exists("cath_data/struc_seq_2048/"+id):
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
            truth = model_ESM.tokenizers.sequence(record)['input_ids']
            seq_split = seq[st[index] - 1: ed[index]]
            #ESM
            ESM_seq = seq.replace("#", "<mask>")
            ESM_protein = ESMProtein(
                sequence=ESM_seq,
            )
            ESM_input = model_ESM.encode(ESM_protein)
            
            #SaProt
            struc_seq = get_struc_seq("foldseek/bin/foldseek", f"cath_data/dompdb/{id}", [id[4:5]], plddt_mask=False, plddt_threshold=70)[id[4:5]][1].lower()
            
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
            #train
            output, esm, saprot, prosst_2048, weights = model.forward(ESM_input, SaProt_input, ProSST_input_ids, ProSST_attention_mask, ProSST_2048_ss_input_ids, start_pos=st[index] - 1)
            pick_pos = [p + 1 for p in pick_pos]
            #for pos in pick_pos:
            #    print(f"{esm[0, pos, truth[pos] - 4]} {saprot[0, pos, truth[pos] - 4]} {prosst_2048[0, pos, truth[pos] - 4]} {output[0, pos, truth[pos] - 4]}")
            #    print(weights[pos])
            preds=torch.zeros(len(pick_pos),20).to(device)
            for i in range(len(pick_pos)):
                preds[i]=output[0, pick_pos[i]]
            labels = torch.tensor([truth[p] - 4 for p in pick_pos]).to(device)
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
    """
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
        protein_chain = ProteinChain.from_pdb("/data/zhangtianhe/proteinMoE/ProteinGym_AF2_structures/" + pdb_file_name)
        ESM_protein = ESMProtein(
            sequence=protein_chain.sequence,
            coordinates=torch.tensor(protein_chain.atom37_positions)
        )
        ESM_protein_tensor = model_ESM.encode(ESM_protein)
        ESM_sequence_tokens = ESM_protein_tensor.sequence
        df['mutated_sequence'] = df['mutant'].apply(
            lambda x: get_mutated_sequence(seq, x))
        pdb_range = mapping_protein_seq_DMS["pdb_range"][mapping_protein_seq_DMS["DMS_id"] == DMS_id].values[0]
        pdb_range = [int(x) for x in pdb_range.split("-")]
        target_seq_split = seq[pdb_range[0] - 1: pdb_range[1]]
        struc_seq = get_struc_seq("foldseek/bin/foldseek", "ProteinGym_AF2_structures/" + pdb_file_name, ["A"], plddt_mask=True, plddt_threshold=70)["A"][1].lower()
        SaProt_seq = "".join([a + b for a, b in zip(target_seq_split, struc_seq)])
        SaProt_tokens = SaProt_tokenizer.tokenize(SaProt_seq)
        with open("ProteinGym_struc_2048/"+prosst_struc_file_name) as file:
            struc_seq_2048 = file.readline()
        struc_seq_2048 = [int(i) for i in struc_seq_2048.split()]
        if len(struc_seq_2048) > len(target_seq_split):
            struc_seq_2048 = struc_seq_2048[:len(target_seq_split)]
        ProSST_2048_ss_input_ids = tokenize_structure_sequence(struc_seq_2048).to(device)
        model_scores = []
        for mut_info in df["mutant"]:
            SaProt_tokens_masked = SaProt_tokens.copy()
            ESM_input_masked = ESM_sequence_tokens.clone()
            ProSST_seq = str(target_seq_split)
            for single in mut_info.split(":"):
                pos = int(single[1:-1]) - pdb_range[0] + 1
                ESM_input_masked[pos] = model_ESM.tokenizers.sequence.mask_token_id
                SaProt_tokens_masked[pos - 1] = "#" + SaProt_tokens_masked[pos - 1][-1]
                ProSST_seq = ProSST_seq[:pos - 1] + "#" + ProSST_seq[pos:]
            ESM_masked_protein_tensor = attr.evolve(ESM_protein_tensor, sequence=ESM_input_masked)
            mask_seq = " ".join(SaProt_tokens_masked)
            SaProt_inputs = SaProt_tokenizer(mask_seq, return_tensors="pt")
            SaProt_inputs = {k: v.to(device) for k, v in SaProt_inputs.items()}
            ProSST_seq = ProSST_seq.replace("#", "<mask>")
            ProSST_tokenized_results = ProSST_tokenizer([ProSST_seq], return_tensors="pt")
            ProSST_input_ids = ProSST_tokenized_results["input_ids"].to(device)
            ProSST_attention_mask = ProSST_tokenized_results["attention_mask"].to(device)
            output, esm, saprot, prosst_2048, weights  = model.forward(ESM_masked_protein_tensor, SaProt_inputs, ProSST_input_ids, ProSST_attention_mask, ProSST_2048_ss_input_ids, start_pos=0)
            model_scores.append(calc(mut_info, target_seq_split, output, pdb_range[0]))
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
