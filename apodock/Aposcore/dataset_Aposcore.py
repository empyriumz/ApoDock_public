import numpy as np
import networkx as nx
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit import Chem
from apodock.Aposcore.common.residue_constants import atom_order
import warnings
from threading import Lock

my_lock = Lock()
RDLogger.DisableLog("rdApp.*")
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")


def one_of_k_encoding_unk(x, allowable_set):
    """Convert x into one-hot encoding with unknown handling"""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(
    mol,
    graph,
    atom_symbols=["C", "N", "O", "S", "F", "P", "Cl", "Br", "I"],
    explicit_H=True,
):
    """Extract atom features from molecule"""
    for atom in mol.GetAtoms():
        results = (
            one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ["Unknown"])
            + one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
            + one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
            + one_of_k_encoding_unk(
                atom.GetHybridization(),
                [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2,
                ],
            )
            + [atom.GetIsAromatic()]
        )
        if explicit_H:
            results = results + one_of_k_encoding_unk(
                atom.GetTotalNumHs(), [0, 1, 2, 3, 4]
            )

        atom_feats = np.array(results).astype(np.float32)
        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))


def get_edge_index(mol, graph):
    """Get edge indices and features from molecule"""
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feats = bond_features(bond)
        graph.add_edge(i, j)
        edge_features.append(feats)
        edge_features.append(feats)
    return torch.stack(edge_features)


def bond_features(bond):
    """Extract bond features"""
    bt = bond.GetBondType()
    fbond = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.IsInRing(),
        bond.GetIsConjugated(),
    ]
    return torch.Tensor(fbond)


def mol2graph(mol):
    """Convert molecule to graph representation"""
    graph = nx.Graph()
    atom_features(mol, graph)
    edge_features = get_edge_index(mol, graph)
    graph = graph.to_directed()
    x = torch.stack([feats["feats"] for _, feats in graph.nodes(data=True)])
    edge_index = torch.stack(
        [torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]
    ).T
    return x, edge_index, edge_features


def get_pro_coord(pro, max_atom_num=24):
    """Get protein coordinates and atom names"""
    coords_pro = []
    coords_main_pro = []
    atom_names = []

    for res in pro:
        atoms = res.get_atoms()
        coords_res = []
        coords_res_main = []
        atom_names_res = []

        for atom in atoms:
            if atom.name in ["N", "CA", "C"]:
                coords_res_main.append(atom.coord)
            coords_res.append(atom.coord)
            atom_names_res.append(atom.name)

        coords_res_main = np.array(coords_res_main)
        coords_res = np.array(coords_res)
        coords_res = np.concatenate(
            [coords_res, np.full((max_atom_num - len(coords_res), 3), np.nan)], axis=0
        )

        atom_names_res = [atom_order[atom] for atom in atom_names_res]
        atom_names_res = np.concatenate(
            [atom_names_res, np.full((max_atom_num - len(atom_names_res),), 37)], axis=0
        )

        coords_main_pro.append(coords_res_main)
        coords_pro.append(coords_res)
        atom_names.append(atom_names_res)

    coords_pro = torch.tensor(coords_pro)
    coords_main_pro = torch.tensor(coords_main_pro)
    atom_names = torch.tensor(atom_names, dtype=torch.long)

    return coords_pro, coords_main_pro, atom_names
