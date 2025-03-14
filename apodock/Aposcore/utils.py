import torch
from apodock.Aposcore.gvp.data import ProteinGraphDataset


three_to_one = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def get_clean_res_list(
    res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None
):
    """Get clean list of residues, filtering out non-standard residues and those without required atoms"""
    res_list = [
        res
        for res in res_list
        if (("N" in res) and ("CA" in res) and ("C" in res) and ("O" in res))
    ]
    clean_res_list = []
    for res in res_list:
        hetero, _, _ = res.full_id[-1]
        if hetero == " ":
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ("CA" in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res["CA"].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list


def get_protein_feature(res_list):
    """Extract protein features using GVP"""
    res_list = [
        res
        for res in res_list
        if (("N" in res) and ("CA" in res) and ("C" in res) and ("O" in res))
    ]
    structure = {}
    structure["name"] = "placeholder"
    structure["seq"] = "".join([three_to_one.get(res.resname) for res in res_list])
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res["N"], res["CA"], res["C"], res["O"]]:
            res_coords.append(list(atom.coord))
        coords.append(res_coords)
    structure["coords"] = coords
    torch.set_num_threads(1)
    dataset = ProteinGraphDataset([structure])
    protein = dataset[0]
    return (
        protein.x,
        protein.seq,
        protein.node_s,
        protein.node_v,
        protein.edge_index,
        protein.edge_s,
        protein.edge_v,
    )


def load_model_dict(model, ckpt):
    """Load model state dict from checkpoint"""
    model.load_state_dict(torch.load(ckpt))
