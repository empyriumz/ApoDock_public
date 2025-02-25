from __future__ import print_function

import numpy as np
import torch
import torch.utils
import prody
from prody import confProDy, writePDB, parsePDB
from torch.nn import functional as F
import apodock.Pack_sc.openfold.np.residue_constants as rc

confProDy(verbosity="none")
from typing import Optional, Union, Tuple


from apodock.Pack_sc.openfold.utils.rigid_utils import Rigid
from apodock.Pack_sc.openfold.utils import feats
from apodock.Pack_sc.openfold.data.data_transforms import atom37_to_torsion_angles
from apodock.Pack_sc.openfold.np.residue_constants import (
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
    restype_atom14_to_rigid_group,
    restype_rigid_group_default_frame,
)
from apodock.Pack_sc.sc_utils import map_mpnn_to_af2_seq


restype_1to3 = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "X": "UNK",
}
restype_str_to_int = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,
}
restype_int_to_str = {
    0: "A",
    1: "C",
    2: "D",
    3: "E",
    4: "F",
    5: "G",
    6: "H",
    7: "I",
    8: "K",
    9: "L",
    10: "M",
    11: "N",
    12: "P",
    13: "Q",
    14: "R",
    15: "S",
    16: "T",
    17: "V",
    18: "W",
    19: "Y",
    20: "X",
}
alphabet = list(restype_str_to_int)

element_list = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mb",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Uut",
    "Fl",
    "Uup",
    "Lv",
    "Uus",
    "Uuo",
]
element_list = [item.upper() for item in element_list]
# element_dict = dict(zip(element_list, range(1,len(element_list))))
element_dict_rev = dict(zip(range(1, len(element_list)), element_list))


def get_seq_rec(S: torch.Tensor, S_pred: torch.Tensor, mask: torch.Tensor):
    """
    S : true sequence shape=[batch, length]
    S_pred : predicted sequence shape=[batch, length]
    mask : mask to compute average over the region shape=[batch, length]

    average : averaged sequence recovery shape=[batch]
    """
    match = S == S_pred
    average = torch.sum(match * mask, dim=-1) / torch.sum(mask, dim=-1)
    return average


def get_score(S: torch.Tensor, log_probs: torch.Tensor, mask: torch.Tensor):
    """
    S : true sequence shape=[batch, length]
    log_probs : predicted sequence shape=[batch, length]
    mask : mask to compute average over the region shape=[batch, length]

    average_loss : averaged categorical cross entropy (CCE) [batch]
    loss_per_resdue : per position CCE [batch, length]
    """
    S_one_hot = torch.nn.functional.one_hot(S, 21)
    loss_per_residue = -(S_one_hot * log_probs).sum(-1)  # [B, L]
    average_loss = torch.sum(loss_per_residue * mask, dim=-1) / (
        torch.sum(mask, dim=-1) + 1e-8
    )
    return average_loss, loss_per_residue


def write_full_PDB(
    save_path: str,
    X: np.ndarray,
    X_m: np.ndarray,
    b_factors: np.ndarray,
    R_idx: np.ndarray,
    chain_letters: np.ndarray,
    S: np.ndarray,
    other_atoms=None,
    icodes=None,
    force_hetatm=False,
):
    """
    save_path : path where the PDB will be written to
    X : protein atom xyz coordinates shape=[length, 14, 3]
    X_m : protein atom mask shape=[length, 14]
    b_factors: shape=[length, 14]
    R_idx: protein residue indices shape=[length]
    chain_letters: protein chain letters shape=[length]
    S : protein amino acid sequence shape=[length]
    other_atoms: other atoms parsed by prody
    icodes: a list of insertion codes for the PDB; e.g. antibody loops
    """
    restype_1to3 = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
        "X": "UNK",
    }
    restype_INTtoSTR = {
        0: "A",
        1: "C",
        2: "D",
        3: "E",
        4: "F",
        5: "G",
        6: "H",
        7: "I",
        8: "K",
        9: "L",
        10: "M",
        11: "N",
        12: "P",
        13: "Q",
        14: "R",
        15: "S",
        16: "T",
        17: "V",
        18: "W",
        19: "Y",
        20: "X",
    }
    restype_name_to_atom14_names = {
        "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
        "ARG": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "NE",
            "CZ",
            "NH1",
            "NH2",
            "",
            "",
            "",
        ],
        "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
        "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
        "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
        "GLN": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "OE1",
            "NE2",
            "",
            "",
            "",
            "",
            "",
        ],
        "GLU": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD",
            "OE1",
            "OE2",
            "",
            "",
            "",
            "",
            "",
        ],
        "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
        "HIS": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "ND1",
            "CD2",
            "CE1",
            "NE2",
            "",
            "",
            "",
            "",
        ],
        "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
        "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
        "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
        "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
        "PHE": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "",
            "",
            "",
        ],
        "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
        "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
        "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
        "TRP": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE2",
            "CE3",
            "NE1",
            "CZ2",
            "CZ3",
            "CH2",
        ],
        "TYR": [
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "OH",
            "",
            "",
        ],
        "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
        "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
    }

    S_str = [
        restype_1to3[AA] for AA in [restype_INTtoSTR[AA] for AA in S]
    ]  # convert to 3 letter code
    # print(S_str)

    X_list = []
    b_factor_list = []
    atom_name_list = []
    element_name_list = []
    residue_name_list = []
    residue_number_list = []
    chain_id_list = []
    icodes_list = []
    for i, AA in enumerate(S_str):
        sel = X_m[i].astype(np.int32) == 1
        total = np.sum(sel)
        tmp = np.array(restype_name_to_atom14_names[AA])[sel]
        X_list.append(X[i][sel])
        b_factor_list.append(b_factors[i][sel])
        atom_name_list.append(tmp)
        element_name_list += [AA[:1] for AA in list(tmp)]
        residue_name_list += total * [AA]
        residue_number_list += total * [R_idx[i]]

        chain_id_list += total * [chain_letters[i]]
        icodes_list += total * [icodes[i]]

    X_stack = np.concatenate(X_list, 0)
    b_factor_stack = np.concatenate(b_factor_list, 0)
    atom_name_stack = np.concatenate(atom_name_list, 0)

    protein = prody.AtomGroup()
    protein.setCoords(X_stack)
    protein.setBetas(b_factor_stack)
    protein.setNames(atom_name_stack)
    protein.setResnames(residue_name_list)
    protein.setElements(element_name_list)
    protein.setOccupancies(np.ones([X_stack.shape[0]]))
    protein.setResnums(residue_number_list)
    protein.setChids(chain_id_list)
    protein.setIcodes(icodes_list)

    if other_atoms:
        other_atoms_g = prody.AtomGroup()
        other_atoms_g.setCoords(other_atoms.getCoords())
        other_atoms_g.setNames(other_atoms.getNames())
        other_atoms_g.setResnames(other_atoms.getResnames())
        other_atoms_g.setElements(other_atoms.getElements())
        other_atoms_g.setOccupancies(other_atoms.getOccupancies())
        other_atoms_g.setResnums(other_atoms.getResnums())
        other_atoms_g.setChids(other_atoms.getChids())
        if force_hetatm:
            other_atoms_g.setFlags("hetatm", other_atoms.getFlags("hetatm"))
        writePDB(save_path, protein + other_atoms_g)
    else:
        writePDB(save_path, protein)


def get_aligned_coordinates(protein_atoms, CA_dict: dict, atom_name: str):
    """
    protein_atoms: prody atom group
    CA_dict: mapping between chain_residue_idx_icodes and integers
    atom_name: atom to be parsed; e.g. CA
    """
    atom_atoms = protein_atoms.select(f"name {atom_name}")

    if atom_atoms != None:
        atom_coords = atom_atoms.getCoords()
        atom_resnums = atom_atoms.getResnums()
        atom_chain_ids = atom_atoms.getChids()
        atom_icodes = atom_atoms.getIcodes()

    atom_coords_ = np.zeros([len(CA_dict), 3], np.float32)
    atom_coords_m = np.zeros([len(CA_dict)], np.int32)
    if atom_atoms != None:
        for i in range(len(atom_resnums)):
            code = atom_chain_ids[i] + "_" + str(atom_resnums[i]) + "_" + atom_icodes[i]
            if code in list(CA_dict):
                atom_coords_[CA_dict[code], :] = atom_coords[i]
                atom_coords_m[CA_dict[code]] = 1
    return atom_coords_, atom_coords_m


def parse_PDB(
    input_path: str,
    device: str = "cpu",
    chains: list = [],
    parse_all_atoms: bool = False,
    parse_atoms_with_zero_occupancy: bool = False,
):
    """
    input_path : path for the input PDB
    device: device for the torch.Tensor
    chains: a list specifying which chains need to be parsed; e.g. ["A", "B"]
    parse_all_atoms: if False parse only N,CA,C,O otherwise all 37 atoms
    parse_atoms_with_zero_occupancy: if True atoms with zero occupancy will be parsed
    """
    element_list = [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mb",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Uut",
        "Fl",
        "Uup",
        "Lv",
        "Uus",
        "Uuo",
    ]
    element_list = [item.upper() for item in element_list]  # 小写转大写
    element_dict = dict(zip(element_list, range(1, len(element_list))))  # 从1开始编号:
    restype_3to1 = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }
    restype_STRtoINT = {
        "A": 0,
        "C": 1,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 5,
        "H": 6,
        "I": 7,
        "K": 8,
        "L": 9,
        "M": 10,
        "N": 11,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "V": 17,
        "W": 18,
        "Y": 19,
        "X": 20,
    }

    atom_order = {
        "N": 0,
        "CA": 1,
        "C": 2,
        "CB": 3,
        "O": 4,
        "CG": 5,
        "CG1": 6,
        "CG2": 7,
        "OG": 8,
        "OG1": 9,
        "SG": 10,
        "CD": 11,
        "CD1": 12,
        "CD2": 13,
        "ND1": 14,
        "ND2": 15,
        "OD1": 16,
        "OD2": 17,
        "SD": 18,
        "CE": 19,
        "CE1": 20,
        "CE2": 21,
        "CE3": 22,
        "NE": 23,
        "NE1": 24,
        "NE2": 25,
        "OE1": 26,
        "OE2": 27,
        "CH2": 28,
        "NH1": 29,
        "NH2": 30,
        "OH": 31,
        "CZ": 32,
        "CZ2": 33,
        "CZ3": 34,
        "NZ": 35,
        "OXT": 36,
    }

    if not parse_all_atoms:
        atom_types = ["N", "CA", "C", "O"]
    else:
        atom_types = [
            "N",
            "CA",
            "C",
            "CB",
            "O",
            "CG",
            "CG1",
            "CG2",
            "OG",
            "OG1",
            "SG",
            "CD",
            "CD1",
            "CD2",
            "ND1",
            "ND2",
            "OD1",
            "OD2",
            "SD",
            "CE",
            "CE1",
            "CE2",
            "CE3",
            "NE",
            "NE1",
            "NE2",
            "OE1",
            "OE2",
            "CH2",
            "NH1",
            "NH2",
            "OH",
            "CZ",
            "CZ2",
            "CZ3",
            "NZ",
        ]

    atoms = parsePDB(input_path)
    if not parse_atoms_with_zero_occupancy:
        atoms = atoms.select("occupancy > 0")
    if chains:
        str_out = ""
        for item in chains:
            str_out += " chain " + item + " or"
        atoms = atoms.select(str_out[1:-3])  # select("chain A or chain B")

    protein_atoms = atoms.select("protein")
    backbone = protein_atoms.select("backbone")
    other_atoms = atoms.select("not protein and not water")
    water_atoms = atoms.select("water")

    CA_atoms = protein_atoms.select("name CA")
    CA_resnums = CA_atoms.getResnums()
    CA_chain_ids = CA_atoms.getChids()
    CA_icodes = CA_atoms.getIcodes()

    CA_dict = {}
    for i in range(len(CA_resnums)):
        code = CA_chain_ids[i] + "_" + str(CA_resnums[i]) + "_" + CA_icodes[i]  # A_1_
        CA_dict[code] = i

    xyz_37 = np.zeros([len(CA_dict), 37, 3], np.float32)
    xyz_37_m = np.zeros([len(CA_dict), 37], np.int32)
    for atom_name in atom_types:
        xyz, xyz_m = get_aligned_coordinates(
            protein_atoms, CA_dict, atom_name
        )  # 获取对齐的原子类型相同的坐标以及mask
        xyz_37[:, atom_order[atom_name], :] = xyz  # 将坐标赋值到xyz_37
        xyz_37_m[:, atom_order[atom_name]] = xyz_m  # 将mask赋值到xyz_37_m

    N = xyz_37[:, atom_order["N"], :]
    CA = xyz_37[:, atom_order["CA"], :]
    C = xyz_37[:, atom_order["C"], :]
    O = xyz_37[:, atom_order["O"], :]

    N_m = xyz_37_m[:, atom_order["N"]]
    CA_m = xyz_37_m[:, atom_order["CA"]]
    C_m = xyz_37_m[:, atom_order["C"]]
    O_m = xyz_37_m[:, atom_order["O"]]

    mask = N_m * CA_m * C_m * O_m  # must all 4 atoms exist

    b = CA - N
    c = C - CA
    a = np.cross(b, c, axis=-1)
    CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

    chain_labels = np.array(CA_atoms.getChindices(), dtype=np.int32)
    R_idx = np.array(CA_resnums, dtype=np.int32)
    S = CA_atoms.getResnames()

    S = [restype_3to1[AA] if AA in list(restype_3to1) else "X" for AA in list(S)]
    S = np.array([restype_STRtoINT[AA] for AA in list(S)], np.int32)

    X = np.concatenate(
        [N[:, None], CA[:, None], C[:, None], O[:, None]], 1
    )  # [length, 4, 3]

    try:
        Y = np.array(other_atoms.getCoords(), dtype=np.float32)
        Y_t = list(other_atoms.getElements())
        Y_t = np.array(
            [
                element_dict[y_t.upper()] if y_t.upper() in element_list else 0
                for y_t in Y_t
            ],
            dtype=np.int32,
        )
        Y_m = (Y_t != 1) * (Y_t != 0)

        Y = Y[Y_m, :]  # only keep atoms with known element
        Y_t = Y_t[Y_m]  # only keep atoms with known element
        Y_m = Y_m[Y_m]
    except:
        Y = np.zeros([1, 3], np.float32)
        Y_t = np.zeros([1], np.int32)
        Y_m = np.zeros([1], np.int32)

    output_dict = {}
    output_dict["X"] = torch.tensor(
        X, device=device, dtype=torch.float32
    )  # [length, 4, 3] 表示每个残基的N,CA,C,O的坐标
    output_dict["mask"] = torch.tensor(
        mask, device=device, dtype=torch.int32
    )  # [length] 表示每个残基是否存在N,CA,C,O
    output_dict["Y"] = torch.tensor(
        Y, device=device, dtype=torch.float32
    )  # [number_of_ligand_atoms, 3] 表示配体的坐标
    output_dict["Y_t"] = torch.tensor(
        Y_t, device=device, dtype=torch.int32
    )  # [number_of_ligand_atoms] 表示配体的元素
    output_dict["Y_m"] = torch.tensor(
        Y_m, device=device, dtype=torch.int32
    )  # [number_of_ligand_atoms] 表示配体的mask

    output_dict["R_idx"] = torch.tensor(
        R_idx, device=device, dtype=torch.int32
    )  # [length] 表示每个残基的编号

    output_dict["chain_labels"] = torch.tensor(
        chain_labels, device=device, dtype=torch.int32
    )  # [length] 表示每个残基的链的编号

    output_dict["chain_letters"] = CA_chain_ids  # 表示每个残基的链的字母 A, B, C, D

    mask_c = []
    chain_list = list(set(output_dict["chain_letters"]))
    chain_list.sort()
    for chain in chain_list:
        mask_c.append(
            torch.tensor(
                [chain == item for item in output_dict["chain_letters"]],
                device=device,
                dtype=bool,
            )
        )

    output_dict["mask_c"] = mask_c
    output_dict["chain_list"] = chain_list

    output_dict["S"] = torch.tensor(
        S, device=device, dtype=torch.int32
    )  # [length] 表示每个残基的氨基酸的

    output_dict["xyz_37"] = torch.tensor(xyz_37, device=device, dtype=torch.float32)
    output_dict["xyz_37_m"] = torch.tensor(xyz_37_m, device=device, dtype=torch.int32)

    return output_dict, backbone, other_atoms, CA_icodes, water_atoms


def get_nearest_neighbours(CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms):
    device = CB.device
    mask_CBY = mask[:, None] * Y_m[None, :]  # [A,B]
    L2_AB = torch.sum((CB[:, None, :] - Y[None, :, :]) ** 2, -1)
    L2_AB = L2_AB * mask_CBY + (1 - mask_CBY) * 1000.0

    nn_idx = torch.argsort(L2_AB, -1)[:, :number_of_ligand_atoms]
    L2_AB_nn = torch.gather(L2_AB, 1, nn_idx)
    D_AB_closest = torch.sqrt(L2_AB_nn[:, 0])

    Y_r = Y[None, :, :].repeat(CB.shape[0], 1, 1)
    Y_t_r = Y_t[None, :].repeat(CB.shape[0], 1)
    Y_m_r = Y_m[None, :].repeat(CB.shape[0], 1)

    Y_tmp = torch.gather(Y_r, 1, nn_idx[:, :, None].repeat(1, 1, 3))
    Y_t_tmp = torch.gather(Y_t_r, 1, nn_idx)
    Y_m_tmp = torch.gather(Y_m_r, 1, nn_idx)

    Y = torch.zeros(
        [CB.shape[0], number_of_ligand_atoms, 3], dtype=torch.float32, device=device
    )
    Y_t = torch.zeros(
        [CB.shape[0], number_of_ligand_atoms], dtype=torch.int32, device=device
    )
    Y_m = torch.zeros(
        [CB.shape[0], number_of_ligand_atoms], dtype=torch.int32, device=device
    )

    num_nn_update = Y_tmp.shape[1]
    Y[:, :num_nn_update] = Y_tmp
    Y_t[:, :num_nn_update] = Y_t_tmp
    Y_m[:, :num_nn_update] = Y_m_tmp

    return Y, Y_t, Y_m, D_AB_closest


def featurize(
    input_dict,
    cutoff_for_score=8.0,
    use_atom_context=True,
    number_of_ligand_atoms=16,
    model_type="protein_mpnn",
):
    output_dict = {}
    if model_type == "ligand_mpnn":
        mask = input_dict["mask"]
        Y = input_dict["Y"]
        Y_t = input_dict["Y_t"]
        Y_m = input_dict["Y_m"]
        N = input_dict["X"][:, 0, :]
        CA = input_dict["X"][:, 1, :]
        C = input_dict["X"][:, 2, :]
        b = CA - N
        c = C - CA
        a = torch.cross(b, c, axis=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA
        Y, Y_t, Y_m, D_XY = get_nearest_neighbours(
            CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms
        )
        mask_XY = (D_XY < cutoff_for_score) * mask * Y_m[:, 0]
        output_dict["mask_XY"] = mask_XY[None,]
        if "side_chain_mask" in list(input_dict):
            output_dict["side_chain_mask"] = input_dict["side_chain_mask"][None,]
        output_dict["Y"] = Y[None,]
        output_dict["Y_t"] = Y_t[None,]
        output_dict["Y_m"] = Y_m[None,]
        if not use_atom_context:
            output_dict["Y_m"] = 0.0 * output_dict["Y_m"]
    elif (
        model_type == "per_residue_label_membrane_mpnn"
        or model_type == "global_label_membrane_mpnn"
    ):
        output_dict["membrane_per_residue_labels"] = input_dict[
            "membrane_per_residue_labels"
        ][
            None,
        ]

    R_idx_list = []
    count = 0
    R_idx_prev = -100000
    for R_idx in list(input_dict["R_idx"]):
        if R_idx_prev == R_idx:
            count += 1
        R_idx_list.append(R_idx + count)
        R_idx_prev = R_idx
    R_idx_renumbered = torch.tensor(R_idx_list, device=R_idx.device)
    output_dict["R_idx"] = R_idx_renumbered[None,]
    output_dict["R_idx_original"] = input_dict["R_idx"][None,]
    # print(input_dict["R_idx"][None,])
    output_dict["chain_labels"] = input_dict["chain_labels"][None,]
    output_dict["S"] = input_dict["S"][None,]
    # output_dict["chain_mask"] = input_dict["chain_mask"][None,]
    output_dict["mask"] = input_dict["mask"][None,]

    output_dict["X"] = input_dict["X"][None,]

    if "xyz_37" in list(input_dict):
        output_dict["xyz_37"] = input_dict["xyz_37"][None,]
        output_dict["xyz_37_m"] = input_dict["xyz_37_m"][None,]

    return output_dict


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
    res_list = [
        res
        for res in res_list
        if (("N" in res) and ("CA" in res) and ("C" in res) and ("O" in res))
    ]
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
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


def get_atom14_coords(X_37, S, atom14_mask, atom37_mask, chi_pred, device):

    xyz37 = torch.zeros(
        [X_37.shape[0], X_37.shape[1], 37, 3], device=device, dtype=torch.float32
    )
    xyz37[:, :, :3] = X_37[:, :, :3]  # N, CA, C
    xyz37[:, :, 4] = X_37[:, :, 3]  # O

    rigids = Rigid.make_transform_from_reference(
        n_xyz=xyz37[:, :, 0, :],
        ca_xyz=xyz37[:, :, 1, :],
        c_xyz=xyz37[:, :, 2, :],
        eps=1e-9,
    )

    # print(rigids.shape)
    # print(torch.sin(chi_pred).shape)
    # print(torch.cos(chi_pred).shape)
    if chi_pred.dim() == 3:
        SC_D_sincos = torch.cat(
            [torch.sin(chi_pred)[..., None], torch.cos(chi_pred)[..., None]], dim=-1
        )

    else:
        SC_D_sincos = chi_pred
    # print(SC_D_sincos.shape)
    # print(rigids.shape)

    S_af2 = torch.argmax(
        torch.nn.functional.one_hot(S, 21).float()
        @ map_mpnn_to_af2_seq.to(device).float(),
        -1,
    )
    S = S_af2
    temp_dict = {
        "aatype": S,
        "all_atom_positions": xyz37,
        "all_atom_mask": atom37_mask,
    }
    torsion_dict = atom37_to_torsion_angles("")(temp_dict)
    # print(torsion_dict["all_atom_positions"].shape)
    # print(S_af2[0][2])
    # print(torsion_dict["torsion_angles_sin_cos"][0][2])
    torsions_noised = torsion_dict["torsion_angles_sin_cos"]  # [batch, length, 7, 2]
    # print(torsions_noised.shape)
    torsions_noised[:, :, 3:] = SC_D_sincos
    # print(torsions_noised.shape)
    # print(S[0][2])
    # print(torsion_dict["torsion_angles_sin_cos"][0][2])
    # print(SC_D_sincos[0][2])
    pred_frames = feats.torsion_angles_to_frames(
        rigids,
        torsions_noised,
        S,
        torch.tensor(restype_rigid_group_default_frame, device=device),
    )
    xyz14_noised = feats.frames_and_literature_positions_to_atom14_pos(
        pred_frames,
        S,
        torch.tensor(restype_rigid_group_default_frame, device=device),
        torch.tensor(restype_atom14_to_rigid_group, device=device),
        torch.tensor(restype_atom14_mask, device=device),
        torch.tensor(restype_atom14_rigid_group_positions, device=device),
    )
    xyz14 = xyz14_noised * atom14_mask[:, :, :, None]
    xyz14[:, :, :3, :] = X_37[:, :, :3, :]
    xyz14[:, :, 3, :] = X_37[:, :, 4, :]

    return xyz14


def get_bb_frames(N: torch.Tensor, CA: torch.Tensor, C: torch.Tensor, fixed=True):
    # N, CA, C = [*, L, 3]
    return Rigid.from_3_points(N, CA, C, fixed=fixed)


def get_atom14_coords_infer(X, S, BB_D, SC_D):
    # x: atom14 coordinates
    # Convert angles to sin/cos
    BB_D_sincos = torch.stack((torch.sin(BB_D), torch.cos(BB_D)), dim=-1)
    SC_D_sincos = torch.stack((torch.sin(SC_D), torch.cos(SC_D)), dim=-1)

    # Get backbone global frames from N, CA, and C
    bb_to_global = get_bb_frames(X[..., 0, :], X[..., 1, :], X[..., 2, :])

    # Concatenate all angles
    angle_agglo = torch.cat([BB_D_sincos, SC_D_sincos], dim=-2)  # [B, L, 7, 2]

    # Get norm of angles
    norm_denom = torch.sqrt(
        torch.clamp(torch.sum(angle_agglo**2, dim=-1, keepdim=True), min=1e-12)
    )

    # Normalize
    normalized_angles = angle_agglo / norm_denom

    # Make default frames
    default_frames = torch.tensor(
        rc.restype_rigid_group_default_frame,
        dtype=torch.float32,
        device=X.device,
        requires_grad=False,
    )

    # Make group ids
    group_idx = torch.tensor(
        rc.restype_atom14_to_rigid_group, device=X.device, requires_grad=False
    )

    # Make atom mask
    atom_mask = torch.tensor(
        rc.restype_atom14_mask,
        dtype=torch.float32,
        device=X.device,
        requires_grad=False,
    )

    # Make literature positions
    lit_positions = torch.tensor(
        rc.restype_atom14_rigid_group_positions,
        dtype=torch.float32,
        device=X.device,
        requires_grad=False,
    )

    # Make all global frames
    all_frames_to_global = feats.torsion_angles_to_frames(
        bb_to_global, normalized_angles, S, default_frames
    )

    # Predict coordinates
    pred_xyz = feats.frames_and_literature_positions_to_atom14_pos(
        all_frames_to_global, S, default_frames, group_idx, atom_mask, lit_positions
    )

    # Replace backbone atoms with input coordinates
    pred_xyz[..., :4, :] = X[..., :4, :]

    return pred_xyz


def chi_angle_to_bin(SC_D, n_chi_bin, chi_mask):
    device = SC_D.device

    # 确保 chi_mask 是布尔类型，且形状匹配
    chi_mask = chi_mask.bool().to(device)

    # 计算角度
    angles = torch.atan2(SC_D[..., 0], SC_D[..., 1])
    zero_tensor = torch.tensor(0.0, device=device)  # 在同一设备上创建零张量
    angles = torch.where(chi_mask, angles, zero_tensor)
    # 调整角度范围
    angles = angles + torch.pi

    # 计算 bin 索引
    SC_D_bin = torch.floor(angles / (2 * torch.pi / n_chi_bin))
    SC_D_bin = torch.where(chi_mask, SC_D_bin, zero_tensor)

    # 计算 bin 偏移量
    SC_D_bin_offset = angles % (2 * torch.pi / n_chi_bin)
    SC_D_bin_offset = torch.where(chi_mask, SC_D_bin_offset, zero_tensor)

    # 确保 bin 索引和偏移量为整数类型
    SC_D_bin = torch.nan_to_num(SC_D_bin).to(torch.int64)

    return SC_D_bin, SC_D_bin_offset


class BlackHole:
    """Dummy object."""

    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


def nll_chi_loss(chi_log_probs, true_chi_bin, sequence, chi_mask, _metric=None):
    """Negative log probabilities for binned chi prediction"""

    # Get which chis are pi periodic
    residue_type_one_hot = F.one_hot(sequence.long(), 21)
    chi_pi_periodic = torch.einsum(
        "...ij, jk->...ik",
        residue_type_one_hot.type(chi_log_probs.dtype),
        chi_mask.new_tensor(np.array(rc.chi_pi_periodic)),
    )

    # Create shifted true chi bin for the pi periodic chis
    n_bins = chi_log_probs.shape[-1]
    shift_val = (n_bins - 1) // 2
    shift = (true_chi_bin >= shift_val) * -shift_val + (
        true_chi_bin < shift_val
    ) * shift_val
    true_chi_bin_shifted = true_chi_bin + shift * chi_pi_periodic

    # NLL loss for shifted and unshifted predictions
    criterion = torch.nn.NLLLoss(reduction="none")
    loss = criterion(
        chi_log_probs.contiguous().view(-1, n_bins),
        true_chi_bin.long().contiguous().view(-1),
    ).view(true_chi_bin.size())
    loss_shifted = criterion(
        chi_log_probs.contiguous().view(-1, n_bins),
        true_chi_bin_shifted.long().contiguous().view(-1),
    ).view(true_chi_bin.size())

    # Determine masked loss and loss average
    loss = torch.minimum(loss, loss_shifted) * chi_mask
    loss_av = torch.sum(loss) / torch.sum(chi_mask)

    if _metric is not None and not isinstance(_metric, BlackHole):
        loss_av = _metric(loss, chi_mask)

    return loss_av


def offset_mse(
    offset_pred, offset_true, mask, n_chi_bin=72, scale_pred=True, _metric=None
):
    if scale_pred:
        offset_pred = (2 * torch.pi / n_chi_bin) * offset_pred

    err = torch.sum(mask * (offset_pred - offset_true) ** 2) / torch.sum(mask)

    if _metric is not None and not isinstance(_metric, BlackHole):
        err = _metric((offset_pred - offset_true) ** 2, mask)

    return err


from typing import Optional, Union, Tuple


def masked_mean(
    mask: torch.Tensor,
    value: torch.Tensor,
    dim: Optional[Union[int, Tuple[int]]] = None,
    eps: float = 1e-4,
) -> torch.Tensor:

    mask = mask.expand(*value.shape)

    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))


def get_renamed_coords(
    X: torch.Tensor, S: torch.Tensor, pseudo_renaming: bool = False
) -> torch.Tensor:
    # Determine which atoms should be swapped.
    if pseudo_renaming:
        atom_renaming_swaps = rc.residue_atom_pseudo_renaming_swaps
    else:
        atom_renaming_swaps = rc.residue_atom_renaming_swaps

    # Rename symmetric atoms
    renamed_X = X.clone()
    for restype in atom_renaming_swaps:
        # Get mask based on restype
        restype_idx = rc.restype_order[rc.restype_3to1[restype]]
        restype_mask = S == restype_idx

        # Swap atom coordinates for restype
        restype_X = renamed_X * restype_mask[..., None, None]
        for atom_pair in atom_renaming_swaps[restype]:
            atom1, atom2 = atom_pair

            atom1_idx, atom2_idx = rc.restype_name_to_atom14_names[restype].index(
                atom1
            ), rc.restype_name_to_atom14_names[restype].index(atom2)

            restype_X[..., atom1_idx, :] = X[..., atom2_idx, :]
            restype_X[..., atom2_idx, :] = X[..., atom1_idx, :]

        # Update full tensor
        restype_X = torch.nan_to_num(restype_X) * restype_mask[..., None, None]
        renamed_X = renamed_X * ~restype_mask[..., None, None] + restype_X

    return renamed_X


def sc_rmsd(decoy_X, true_X, S, X_mask, residue_mask, _metric=None, use_sqrt=False):
    # Compute atom deviation based on original coordinates
    # print("pred:",decoy_X[0][0])
    # print("true",true_X[0][0])
    atom_deviation = torch.sum(torch.square(decoy_X - true_X), dim=-1)

    # Compute atom deviation based on alternative coordinates
    true_renamed_X = get_renamed_coords(true_X, S)
    renamed_atom_deviation = torch.sum(torch.square(decoy_X - true_renamed_X), dim=-1)

    # Get atom mask including backbone atoms
    atom_mask = X_mask * residue_mask[..., None]
    atom_mask[..., :4] = 0.0

    # Compute RMSD based on original and alternative coordinates
    rmsd_og = masked_mean(atom_mask, atom_deviation, -1)
    rmsd_renamed = masked_mean(atom_mask, renamed_atom_deviation, -1)
    if use_sqrt:
        rmsd_og = torch.sqrt(rmsd_og)
        rmsd_renamed = torch.sqrt(rmsd_renamed)
    rmsd = torch.minimum(rmsd_og, rmsd_renamed)
    # print(rmsd.shape)
    if _metric is not None and not isinstance(_metric, BlackHole):
        mse = _metric(rmsd)
    else:
        return rmsd.mean()

    return mse


Array = Union[np.ndarray, torch.Tensor]


def robust_norm(
    array: Array, axis: int = -1, l_norm: float = 2, eps: float = 1e-8
) -> Array:
    """Computes robust l-norm of vectors.

    Args:
        array (Array): Array containing vectors to compute norm.
        axis (int, optional): Axis of array to norm. Defaults to -1.
        l_norm (float, optional): Norm-type to perform. Defaults to 2.
        eps (float, optional): Epsilon for robust norm computation. Defaults to 1e-8.

    Returns:
        Array: Norm of axis of array
    """
    if isinstance(array, np.ndarray):
        return (np.sum(array**l_norm, axis=axis) + eps) ** (1 / l_norm)
    else:
        return (torch.sum(array**l_norm, dim=axis) + eps) ** (1 / l_norm)


def robust_normalize(
    array: Array, axis: int = -1, l_norm: float = 2, eps: float = 1e-8
) -> Array:
    """Computes robust l-normalization of vectors.

    Args:
        array (Array): Array containing vectors to normalize.
        axis (int, optional): Axis of array to normalize. Defaults to -1.
        l_norm (float, optional): Norm-type to perform. Defaults to 2.
        eps (float, optional): Epsilon for robust norma computation. Defaults to 1e-8.

    Returns:
        Array: Normalized array
    """
    if isinstance(array, np.ndarray):
        return array / np.expand_dims(
            robust_norm(array, axis=axis, l_norm=l_norm, eps=eps), axis=axis
        )
    else:
        return array / robust_norm(array, axis=axis, l_norm=l_norm, eps=eps).unsqueeze(
            axis
        )


def _calc_dihedrals(atom_positions: torch.Tensor, eps=1e-6) -> torch.Tensor:
    # Unit vectors
    uvecs = robust_normalize(
        atom_positions[..., 1:, :] - atom_positions[..., :-1, :], eps=eps
    )
    uvec_2 = uvecs[..., :-2, :]
    uvec_1 = uvecs[..., 1:-1, :]
    uvec_0 = uvecs[..., 2:, :]

    # Normals
    nvec_2 = robust_normalize(torch.cross(uvec_2, uvec_1, dim=-1), eps=eps)
    nvec_1 = robust_normalize(torch.cross(uvec_1, uvec_0, dim=-1), eps=eps)

    # Angle between normals
    cos_dihedral = torch.sum(nvec_2 * nvec_1, dim=-1)
    cos_dihedral = torch.clamp(cos_dihedral, -1 + eps, 1 - eps)
    # print(torch.any(torch.isnan(cos_dihedral)))
    dihedral = torch.sign(torch.sum(uvec_2 * nvec_1, dim=-1)) * torch.acos(cos_dihedral)

    return dihedral


def calc_sc_dihedrals(
    atom_positions: torch.Tensor, aatype: torch.Tensor, return_mask: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # Get atom indicies for atoms that make up chi angles and chi mask
    chi_atom_indices = torch.tensor(chi_atom_indices_atom14).to(aatype.device)[aatype]
    chi_mask = torch.tensor(chi_mask_atom14).to(aatype.device)[aatype]

    # Get coordinates for chi atoms
    chi_atom_positions = torch.gather(
        atom_positions,
        -2,
        chi_atom_indices[..., None].expand(*chi_atom_indices.shape, 3).long(),
    )
    sc_dihedrals = _calc_dihedrals(chi_atom_positions)

    # Chi angles that are missing an atom will be NaN, so turn all those to 0.
    sc_dihedrals = torch.nan_to_num(sc_dihedrals)

    # Mask nonexistent chis based on sequence.
    sc_dihedrals = sc_dihedrals * chi_mask
    sc_dihedrals_mask = (sc_dihedrals != 0.0).to(torch.float32)

    if return_mask:
        return sc_dihedrals, sc_dihedrals_mask
    else:
        return sc_dihedrals


def calc_bb_dihedrals(
    atom_positions: Array,
    residue_index: Optional[Array] = None,
    use_pre_omega: bool = True,
    return_mask: bool = True,
) -> Union[Array, Tuple[Array, Array]]:

    # Get backbone coordinates (and reshape). First 3 coordinates are N, CA, C
    bb_atom_positions = atom_positions[:, :3].reshape((3 * atom_positions.shape[0], 3))

    # Get backbone dihedrals
    bb_dihedrals = _calc_dihedrals(bb_atom_positions)
    if isinstance(atom_positions, np.ndarray):
        bb_dihedrals = np.pad(
            bb_dihedrals, [(1, 2)], constant_values=np.nan
        )  # Add empty phi[0], psi[-1], and omega[-1]
        bb_dihedrals = bb_dihedrals.reshape((atom_positions.shape[0], 3))

        # Get mask based on residue_index
        bb_dihedrals_mask = np.ones_like(bb_dihedrals)
        if residue_index is not None:
            assert type(atom_positions) == type(residue_index)
            pre_mask = np.concatenate(
                (
                    [0.0],
                    (residue_index[1:] - 1 == residue_index[:-1]).astype(np.float32),
                ),
                axis=-1,
            )
            post_mask = np.concatenate(
                (
                    (residue_index[:-1] + 1 == residue_index[1:]).astype(np.float32),
                    [0.0],
                ),
                axis=-1,
            )
            bb_dihedrals_mask = np.stack((pre_mask, post_mask, post_mask), axis=-1)

        if use_pre_omega:
            # Move omegas such that they're "pre-omegas" and reorder dihedrals
            bb_dihedrals[:, 2] = np.concatenate(
                ([np.nan], bb_dihedrals[:-1, 2]), axis=-1
            )
            bb_dihedrals[:, [0, 1, 2]] = bb_dihedrals[:, [2, 0, 1]]
            bb_dihedrals_mask[:, 1] = bb_dihedrals_mask[:, 0]

        # Update dihedral_mask
        bb_dihedrals_mask = bb_dihedrals_mask * np.isfinite(bb_dihedrals).astype(
            np.float32
        )
    else:
        bb_dihedrals = F.pad(
            bb_dihedrals, [1, 2], value=torch.nan
        )  # Add empty phi[0], psi[-1], and omega[-1]
        bb_dihedrals = bb_dihedrals.reshape((atom_positions.shape[0], 3))

        # Get mask based on residue_index
        bb_dihedrals_mask = torch.ones_like(bb_dihedrals)
        if residue_index is not None:
            assert type(atom_positions) == type(residue_index)
            pre_mask = torch.cat(
                (
                    torch.tensor([0.0]),
                    (residue_index[1:] - 1 == residue_index[:-1]).to(torch.float32),
                ),
                dim=-1,
            )
            post_mask = torch.cat(
                (
                    (residue_index[:-1] + 1 == residue_index[1:]).to(torch.float32),
                    torch.tensor([0.0]),
                ),
                dim=-1,
            )
            bb_dihedrals_mask = torch.stack((pre_mask, post_mask, post_mask), axis=-1)

        if use_pre_omega:
            # Move omegas such that they're "pre-omegas" and reorder dihedrals
            bb_dihedrals[:, 2] = torch.cat(
                (torch.tensor([torch.nan]), bb_dihedrals[:-1, 2]), dim=-1
            )
            bb_dihedrals[:, [0, 1, 2]] = bb_dihedrals[:, [2, 0, 1]]
            bb_dihedrals_mask[:, 1] = bb_dihedrals_mask[:, 0]

        # Update dihedral_mask
        bb_dihedrals_mask = bb_dihedrals_mask * torch.isfinite(bb_dihedrals).to(
            torch.float32
        )
    if return_mask:
        return bb_dihedrals, bb_dihedrals_mask
    else:
        return bb_dihedrals


def wrapped_chi_angle(CH_pred, CH_true):

    # Determine which rotamers are correct via wrapped from negative end
    dist_neg_edge = torch.abs(CH_pred - -180.0)
    close_to_neg_edge = dist_neg_edge <= 20.0
    wrapped_dist = 20.0 - dist_neg_edge
    wrapped_rot_neg = (CH_true > 180.0 - wrapped_dist) * close_to_neg_edge

    # Determine which rotamers are correct via wrapping from positive end
    dist_pos_edge = torch.abs(CH_pred - 180.0)
    close_to_pos_edge = dist_pos_edge <= 20.0
    wrapped_dist = 20.0 - dist_pos_edge
    wrapped_rot_pos = (CH_true < -180.0 + wrapped_dist) * close_to_pos_edge

    return wrapped_rot_neg + wrapped_rot_pos


def pi_periodic_rotamer(CH_pred, CH_true, S):

    # Get which chis are pi periodic
    residue_type_one_hot = F.one_hot(S, 21)
    chi_pi_periodic = torch.einsum(
        "...ij, jk->...ik",
        residue_type_one_hot.type(CH_pred.dtype),
        CH_pred.new_tensor(np.array(rc.chi_pi_periodic)),
    )

    # Shift for the predicted chis
    shift = (CH_pred < CH_true) * 180.0 + (CH_pred > CH_true) * -180.0

    return (torch.abs(CH_pred + shift - CH_true) <= 20.0) * chi_pi_periodic


def rotamer_recovery_from_coords(
    S,
    true_SC_D,
    pred_X,
    residue_mask,
    SC_D_mask,
    return_raw=False,
    return_chis=False,
    exclude_AG=True,
    _metric=None,
):

    # Compute true and predicted chi dihedrals (in degrees)
    CH_true = true_SC_D * 180.0 / torch.pi
    CH_pred = (
        torch.nan_to_num(calc_sc_dihedrals(pred_X, S, return_mask=False))
        * 180.0
        / torch.pi
    )
    # Determine correct chis based on angle difference
    angle_diff_chis = (torch.abs(CH_true - CH_pred) <= 20.0) * SC_D_mask  # [B, L, 4]

    # Determine correct chis based on non-existant chis
    nonexistent_chis = 1.0 - SC_D_mask  # [B, L, 4]

    # Determine correct chis based on wrapping of dihedral angles around -180. and 180.
    wrapped_chis = wrapped_chi_angle(CH_pred, CH_true) * SC_D_mask  # [B, L, 4]

    # Determine correct chis based on periodic chis
    periodic_chis = pi_periodic_rotamer(CH_pred, CH_true, S) * SC_D_mask  # [B, L, 4]

    # Sum to determine correct chis
    correct_chis = (
        angle_diff_chis + nonexistent_chis + wrapped_chis + periodic_chis
    )  # [B, L, 4]

    # Determine correct rotamers based on all correct chi
    correct_rotamer = torch.sum(correct_chis, dim=-1) == 4  # [B, L]

    # Exclude Ala and Gly
    if exclude_AG:
        ala_mask = (S == rc.restype_order["A"]).float()  # [B, L]
        gly_mask = (S == rc.restype_order["G"]).float()  # [B, L]
        residue_mask = residue_mask * (1.0 - ala_mask) * (1.0 - gly_mask)  # [B, L]

    # Determine average number of correct rotamers for each chain (depending on that chains length)
    if _metric is not None and not isinstance(_metric, BlackHole):
        rr = _metric(correct_rotamer, residue_mask)
    else:
        rr = torch.sum(correct_rotamer * residue_mask, dim=-1) / torch.sum(
            residue_mask, dim=-1
        )

    if return_raw:
        return correct_rotamer
    if return_chis:
        return correct_chis
    else:
        return torch.mean(rr)


class BlackHole:
    """Dummy object."""

    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]


def _get_chi_atom_indices_and_mask(use_atom14=True):
    chi_atom_indices = []
    chi_mask = []
    for res_name in restypes:
        res_name = restype_1to3[res_name]
        res_chi_angles = rc.chi_angles_atoms[res_name]

        # Chi mask where 1 for existing chi angle and 0 for nonexistent chi angle
        chi_mask.append([1] * len(res_chi_angles) + [0] * (4 - len(res_chi_angles)))

        # All unique atoms for chi angles
        atoms = [atom for chi in res_chi_angles for atom in chi]
        atoms = sorted(set(atoms), key=lambda x: atoms.index(x))

        # Indices of unique atoms
        if use_atom14:
            atom_indices = [
                rc.restype_name_to_atom14_names[res_name].index(atom) for atom in atoms
            ]
        else:
            atom_indices = [rc.atom_order[atom] for atom in atoms]

        for _ in range(7 - len(atom_indices)):
            atom_indices.append(0)

        chi_atom_indices.append(atom_indices)

    # Update for unknown restype
    chi_atom_indices.append([0] * 7)
    chi_mask.append([0] * 4)

    return chi_atom_indices, chi_mask


chi_atom_indices_atom14, chi_mask_atom14 = _get_chi_atom_indices_and_mask(
    use_atom14=True
)
chi_atom_indices_atom37, chi_mask_atom37 = _get_chi_atom_indices_and_mask(
    use_atom14=False
)


def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))
