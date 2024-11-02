from Bio import PDB
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
# from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from collections import OrderedDict
import pandas as pd

# login("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")
# client = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda") # or "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
client = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device)

def get_esm3_embbeding(sequence):
    print(len(sequence))
    protein = ESMProtein(sequence=(f"{sequence}"))
    # print(protein)
    protein_tensor = client.encode(protein)
    output = client.forward_and_sample(
        protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
    )
    # print(dir(output))
    # 去掉首位的特殊字符
    print(output.per_residue_embedding.shape)
    output = output.per_residue_embedding[1:-1]  # [1:-1] to remove the special tokens in auto-regressive models
    return output


aa_dict = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def extract_sequences_from_pdb(pdb_file):
    parser = PDB.PDBParser()
    structure = parser.get_structure('PDB', pdb_file)
    sequences = OrderedDict()
    res_info = []
    for model in structure:
        for chain in model:
            seq = []
            for residue in chain.get_residues():
                resname = residue.get_resname()
                if resname in aa_dict:  # Convert using the dictionary
                    seq.append(aa_dict[resname])
                    res_info.append((chain.id, residue.id[1]))
            sequences[chain.id] = ''.join(seq)

    return sequences, res_info


def get_pocket_esm3_embedding(full_pdb, pocket_pdb):
    sequences, res_info = extract_sequences_from_pdb(full_pdb)
    # print(len(res_info))
    esm3_embeddings = []
    for i in sequences.values():
        esm3_embeddings.append(get_esm3_embbeding(i))
    # print(esm3_embeddings[0][0], esm3_embeddings[1][0])
    # 判断 res_info 中每个氨基酸的顺序和 esm3_embeddings 中的顺序是否一致
    # print(res_info  )
    assert len(list(sequences.values())[0]) ==  esm3_embeddings[0].shape[0]
    esm3_embeddings = torch.cat(esm3_embeddings, dim=0)

    _, res_info_pocket = extract_sequences_from_pdb(pocket_pdb)

    pocket_embedding_List = []
    res_info_dict = { chain_id + str(res_id): embedding for (chain_id, res_id), embedding in zip(res_info, esm3_embeddings)}
    for chain_id, res_id in res_info_pocket:
        embedding =  res_info_dict.get(chain_id + str(res_id))

        if embedding is None or torch.isnan(embedding).any():
            error_msg = f"missing or nan embedding for {chain_id + str(res_id)} in {full_pdb}"
            raise ValueError(error_msg)
        pocket_embedding_List.append(embedding)

    pocket_embedding = torch.stack(pocket_embedding_List, dim=0)
    
    assert pocket_embedding.shape[0] == len(res_info_pocket)

    return pocket_embedding



def get_esm3_embbeding_pdbbind(datadir, data_df, pocket_dist = 10):
    for i ,row in data_df.iterrows():
        print(row['pdb'])
        if os.path.exists(os.path.join(datadir, "v2020-other-PL", row['pdb'], f"Pocket_clean_{pocket_dist}A_esm3.pt")):
            print(f"skip {row['pdb']}")
            continue
        pdb = os.path.join(datadir, "protein_remove_extra_chains_10A", row['pdb'] + '_protein.pdb')
        pdb_pocket = os.path.join(datadir, "v2020-other-PL", row['pdb'], f"Pocket_clean_{pocket_dist}A.pdb")
        embeddings = get_pocket_esm3_embedding(pdb, pdb_pocket)
        torch.save(embeddings, os.path.join(datadir, "v2020-other-PL", row['pdb'], f"Pocket_clean_{pocket_dist}A_esm3.pt"))
        print(f"save {row['pdb']} esm3 embedding")





if __name__ == "__main__":
    pdb = "../test/1a0q/1a0q_protein.pdb"
    pdb_pocket = "../test/1a0q/Pocket_clean_1a0q.pdb"
    sequences = extract_sequences_from_pdb(pdb)
    print(sequences[1])
    embeddings = get_pocket_esm3_embedding(pdb, pdb_pocket)
    print(embeddings.shape)

    # data_root = "../../PLmodel/supervised/data/pdbbind"
    # data_df = pd.read_csv(os.path.join(data_root, "data.csv"))

    # get_esm3_embbeding_pdbbind(data_root, data_df, pocket_dist = 10)
    # print("done")
