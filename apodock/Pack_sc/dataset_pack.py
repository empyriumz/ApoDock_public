import numpy as np
import multiprocessing
import networkx as nx
import torch
from rdkit import Chem
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain
from apodock.Pack_sc.data_utils import (
    parse_PDB,
    featurize,
    calc_bb_dihedrals,
)

multiprocessing.set_start_method("spawn", force=True)
from apodock.Pack_sc.model_utils import ProteinMPNN


def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(
    mol,
    graph,
    atom_symbols=["C", "N", "O", "S", "F", "P", "Cl", "Br", "I"],
    explicit_H=True,
):

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
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(
                atom.GetTotalNumHs(), [0, 1, 2, 3, 4]
            )

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))


def get_edge_index(mol, graph):
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
    # 返回一个[1,6]的张量，表示一键的各种信息是否存在
    bt = bond.GetBondType()  # 获取键的类型
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
    graph = nx.Graph()
    atom_features(mol, graph)
    edge_features = get_edge_index(mol, graph)
    graph = graph.to_directed()
    x = torch.stack([feats["feats"] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack(
        [torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]
    ).T
    degrees = torch.tensor([graph.degree(node) for node in graph.nodes()])

    return x, edge_index, edge_features, degrees


def get_protein_mpnn_encoder_features(
    pdb, chain_id, pocket_residue_ids, checkpoint_path, device="cpu"
):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # atom_context_num = 1
    # ligand_mpnn_use_side_chain_context = 0
    # k_neighbors = checkpoint["num_edges"]
    model = ProteinMPNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    output_dict, _, _, _, _ = parse_PDB(
        pdb,
        parse_all_atoms=True,
        parse_atoms_with_zero_occupancy=True,
    )
    features_dict = featurize(output_dict)

    h_V, _, _ = model.encode(features_dict)
    # 清空显卡缓存
    torch.cuda.empty_cache()
    full_pdb_features = h_V
    # 储存元组形式的链ID, 残基ID, 以及残基的特征
    full_chain_letters = output_dict["chain_letters"]
    full_R_idx = output_dict["R_idx"].numpy()
    # print(full_R_idx)

    full_residue_features = {
        key: value
        for key, value in zip(zip(full_chain_letters, full_R_idx), full_pdb_features[0])
    }
    # for chain, res_id, feature in zip(output_dict["chain_letters"], output_dict["R_idx"], full_pdb_features[0]):
    #     full_residue_features.append((chain, res_id, feature))
    # print(full_residue_features.keys())

    # 根据口袋残基的链ID和残基ID提取对应的特征
    pocket_features_list = []
    for chain, res_id in zip(chain_id, pocket_residue_ids):
        # print((chain, res_id.numpy().item()))
        features = full_residue_features.get((chain, res_id.numpy().item()))
        if features is None or torch.isnan(features).any():
            # 如果没有找到tensor或者tensor包含NaN，抛出异常
            error_msg = (
                f"Missing or invalid tensor for residue {res_id} in chain {chain}."
            )
            raise ValueError(error_msg)
        pocket_features_list.append(features)

    # 将列表转换为张量
    pocket_features = torch.stack(pocket_features_list)
    #
    # 去掉梯度
    pocket_features = pocket_features.detach()
    # print(pocket_features)
    return pocket_features


def res2pdb(res_list, pdbid, save_path):
    # 创建一个新的结构对象
    new_structure = Structure.Structure("New_Structure")
    model = Model.Model(0)
    new_structure.add(model)
    # 创建链的字典来存储不同链
    chains = {}

    for res in res_list:
        chain_id = res.get_full_id()[2]  # 获取原始链ID
        if chain_id not in chains:
            chain = Chain.Chain(chain_id)
            model.add(chain)
            chains[chain_id] = chain
        chains[chain_id].add(res)

    # 创建PDBIO对象并写入文件
    io = PDBIO()
    io.set_structure(new_structure)
    io.save(save_path)


def get_backbone_atoms(structure):
    """从BioPython的结构对象提取主链原子的坐标，以及残基的信息"""
    coords = []
    res_info = []
    for model in structure:
        for chain in model:
            for residue in chain:
                try:
                    # 仅提取有完整主链原子的残基
                    backbone = [
                        residue["N"].coord,
                        residue["CA"].coord,
                        residue["C"].coord,
                        residue["O"].coord,
                    ]
                    coords.append(backbone)
                    res_info.append((chain.id, residue.id[1]))  # 存储链ID和残基序号
                except KeyError:
                    continue
    return np.array(coords), res_info


def extract_pocket_dihedrals(
    dihedrals, res_info, pocket_residue_ids_list, chain_ids, pdbid
):
    """
    从整个蛋白质的二面角中提取与口袋残基顺序相匹配的二面角信息，并确保无NaN值。
    如果存在NaN，抛出异常并报告问题的残基。
    """
    pocket_dihedral_list = []
    res_info_dict = {
        (chain, res_id): dihedral
        for (chain, res_id), dihedral in zip(res_info, dihedrals)
    }
    # print(res_info_dict)
    # 按照输入列表的顺序提取二面角
    for chain, res_id in zip(chain_ids, pocket_residue_ids_list):
        # print((str(chain), res_id.numpy().item()))
        dihedral = res_info_dict.get((str(chain), res_id.numpy().item()))
        if dihedral is None:  # or torch.isnan(dihedral).any()
            # 如果没有找到二面角   #或者二面角包含NaN，抛出异常
            error_msg = f"Missing or invalid dihedral for residue {res_id} in chain {chain} of {pdbid}."
            raise ValueError(error_msg)
        pocket_dihedral_list.append(dihedral)

    # 将列表转换为张量
    pocket_dihedrals = torch.stack(pocket_dihedral_list)
    return pocket_dihedrals


def Pocket_BB_D(pdb_file, chain_id, pocket_residue_ids):
    pdbid = pdb_file.split("/")[-1].split("_")[0]
    chain_id = list(chain_id)
    # print(chain_id)
    pocket_residue_ids = pocket_residue_ids
    # print(pocket_residue_ids)
    parser = PDBParser()
    structure = parser.get_structure("Protein", pdb_file)
    coords, res_info = get_backbone_atoms(structure)
    coords = torch.tensor(coords)
    # print(coords.shape)
    dihedrals, _ = calc_bb_dihedrals(coords)

    pocket_dihedrals = extract_pocket_dihedrals(
        dihedrals, res_info, pocket_residue_ids, chain_id, pdbid
    )
    return pocket_dihedrals
