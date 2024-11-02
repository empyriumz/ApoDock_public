import os
import pandas as pd
import numpy as np
import pickle
import multiprocessing
from itertools import repeat
import networkx as nx
import torch 
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain
from data_utils import parse_PDB, get_clean_res_list,featurize, calc_bb_dihedrals, load_model_dict
from sc_utils import make_torsion_features
multiprocessing.set_start_method('spawn', force=True)
# from openfold.data.data_transforms import atom37_to_torsion_angles, make_atom14_masks
# from openfold.np.residue_constants import (
#     restype_atom14_mask,
#     restype_atom14_rigid_group_positions,
#     restype_atom14_to_rigid_group,
#     restype_rigid_group_default_frame,
# )
# from openfold.utils import feats
# from openfold.utils.rigid_utils import Rigid
from torch_geometric.data import Data, Batch
from model_utils import ProteinMPNN

def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):

    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

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
    bt = bond.GetBondType() # 获取键的类型
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
             bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing(),bond.GetIsConjugated()]
    return torch.Tensor(fbond)

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    edge_features = get_edge_index(mol, graph)
    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T
    degrees = torch.tensor([graph.degree(node) for node in graph.nodes()])

    return x, edge_index, edge_features, degrees

# def extract_pocket_features(pocket, ligand, pocket_dis=10):

def get_protein_mpnn_encoder_features(pdb, chain_id, pocket_residue_ids, checkpoint_path, device='cpu'):
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
        parse_all_atoms = True,
        parse_atoms_with_zero_occupancy = True,
    )
    # print(type(output_dict["S"]))
    # for key, value in output_dict.items():
    #     print(key, type(value))
    # output_dict = {key: value.unsqueeze(0) for key, value in output_dict.items() if key != "chain_letters" or key != "chain_list"}
    features_dict = featurize(output_dict)
    # print(len(output_dict["chain_letters"]))
    # print(len(output_dict["R_idx"]))

    

    h_V, _, _ = model.encode(features_dict)
    # 清空显卡缓存
    torch.cuda.empty_cache()
    full_pdb_features = h_V
    # 储存元组形式的链ID, 残基ID, 以及残基的特征
    full_chain_letters = output_dict["chain_letters"]
    full_R_idx = output_dict["R_idx"].numpy()
    # print(full_R_idx)

    full_residue_features = {key: value for key, value in zip(zip(full_chain_letters, full_R_idx), full_pdb_features[0])}
    # for chain, res_id, feature in zip(output_dict["chain_letters"], output_dict["R_idx"], full_pdb_features[0]):
    #     full_residue_features.append((chain, res_id, feature))
    # print(full_residue_features.keys())

    # 根据口袋残基的链ID和残基ID提取对应的特征
    pocket_features_list = []
    for chain, res_id in zip(chain_id, pocket_residue_ids):
        # print((chain, res_id.numpy().item()))
        features =  full_residue_features.get((chain, res_id.numpy().item()))
        if features is None or torch.isnan(features).any():
            # 如果没有找到tensor或者tensor包含NaN，抛出异常
            error_msg = f"Missing or invalid tensor for residue {res_id} in chain {chain}."
            raise ValueError(error_msg)
        pocket_features_list.append(features)
    
    # 将列表转换为张量
    pocket_features = torch.stack(pocket_features_list)
    # 
    # 去掉梯度
    pocket_features = pocket_features.detach()
    # print(pocket_features)
    return pocket_features







    # print(h_V.shape)
    # print(h_E.shape)
    # print(E_idx.shape)

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


def get_pro_coord(pro,max_atom_num=24):
    # 获取残基的所有原子的坐标，包括side chain
    coords_pro = []
    coords_main_pro = [] # 只包含主链N,CA,C原子的坐标
    for res in pro:
        # print(len(res.get_atoms()))
        # break
        atoms = res.get_atoms()
        # print(len(atoms))
        # print(dir(atoms))
        # break
        coords_res = []
        coords_res_main = []
        for atom in atoms:
            # print(atom.name)
            # break
            if atom.name in ['N', 'CA', 'C']:
                coords_res_main.append(atom.coord)
            coords_res.append(atom.coord)
            
        coords_res_main = np.array(coords_res_main)
        coords_res = np.array(coords_res)
        # print(len(coords_res))
        coords_res = np.concatenate([coords_res, np.full((max_atom_num - len(coords_res), 3),np.nan)], axis=0)
        # print(coords.shape)
        # break
        coords_main_pro.append(coords_res_main)
        coords_pro.append(coords_res) 
    # print(coords_pro[0].shape)
        
    coords_pro = torch.tensor(coords_pro)
    coords_main_pro = torch.tensor(coords_main_pro)
    # print(coords_pro.shape)
    # print(coords_main_pro.shape)
    return coords_pro, coords_main_pro

def get_backbone_atoms(structure):
    """从BioPython的结构对象提取主链原子的坐标，以及残基的信息"""
    coords = []
    res_info = []
    for model in structure:
        for chain in model:
            for residue in chain:
                try:
                    # 仅提取有完整主链原子的残基
                    backbone = [residue['N'].coord, residue['CA'].coord, residue['C'].coord, residue['O'].coord]
                    coords.append(backbone)
                    res_info.append((chain.id, residue.id[1]))  # 存储链ID和残基序号
                except KeyError:
                    continue
    return np.array(coords), res_info


def extract_pocket_dihedrals(dihedrals, res_info, pocket_residue_ids_list, chain_ids, pdbid):
    """
    从整个蛋白质的二面角中提取与口袋残基顺序相匹配的二面角信息，并确保无NaN值。
    如果存在NaN，抛出异常并报告问题的残基。
    """
    pocket_dihedral_list = []
    res_info_dict = {(chain, res_id): dihedral for (chain, res_id), dihedral in zip(res_info, dihedrals)}
    # print(res_info_dict)
    # 按照输入列表的顺序提取二面角
    for chain, res_id in zip(chain_ids, pocket_residue_ids_list):
        # print((str(chain), res_id.numpy().item()))
        dihedral = res_info_dict.get((str(chain), res_id.numpy().item()))
        if dihedral is None : #or torch.isnan(dihedral).any()
            # 如果没有找到二面角   #或者二面角包含NaN，抛出异常
            error_msg = f"Missing or invalid dihedral for residue {res_id} in chain {chain} of {pdbid}."
            raise ValueError(error_msg)
        pocket_dihedral_list.append(dihedral)
    
    # 将列表转换为张量
    pocket_dihedrals = torch.stack(pocket_dihedral_list)
    return pocket_dihedrals


def Pocket_BB_D(pdb_file, chain_id, pocket_residue_ids):
    pdbid = pdb_file.split('/')[-1].split('_')[0]
    chain_id = list(chain_id)
    # print(chain_id)
    pocket_residue_ids = pocket_residue_ids
    # print(pocket_residue_ids)
    parser = PDBParser()
    structure = parser.get_structure('Protein', pdb_file)
    coords, res_info = get_backbone_atoms(structure)
    coords = torch.tensor(coords)
    # print(coords.shape)
    dihedrals,_ = calc_bb_dihedrals(coords)
    # print(dihedrals.shape)
    # conver dihedrals to numpy
    # dihedrals = dihedrals.numpy()
    pocket_dihedrals = extract_pocket_dihedrals(dihedrals, res_info, pocket_residue_ids, chain_id,pdbid)
    # print(pocket_dihedrals.shape)

    return pocket_dihedrals



def mols2graphs(complex_path,pdbid,save_path_l,save_path_aa, ref_pocket, pocket_dis=10):
    print(pdbid)
    with open(complex_path, 'rb') as f:
        ligand, pocket = pickle.load(f)
    atom_num_l = ligand.GetNumAtoms()
    c_size_l = atom_num_l
    x_l, edge_index_l,edge_features_l, _ = mol2graph(ligand)
    pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
    data_l = Data(
                x=x_l,
                edge_index=edge_index_l,
                edge_attr=edge_features_l,
                pos_l = pos_l,             
                )
    # print(x_l.shape)
    data_l.__setitem__('c_size', torch.LongTensor([c_size_l]))


    parser = PDBParser(QUIET=True)
    
    if ref_pocket:
        pocket_id = pdbid.split('_')[0]
        pocket_pdb = f"./data/pdbbind/v2020-other-PL/{pocket_id}/Pocket_{pocket_dis}A.pdb"
    

    pocket_pdb_dir = os.path.join(
                                 os.path.dirname(complex_path),
                                f'Pocket_{pocket_dis}A.pdb'
                                )
    res_list = get_clean_res_list(parser.get_structure(pdbid, pocket_pdb_dir).get_residues(), verbose=False, ensure_ca_exist=True)
 
    # save biopython res_list as pdb file
    # print(pdbid, len(res_list))
     # make sure the res_list is not empty print(pdbid)
    
    if len(res_list) == 0:
        raise ValueError(f"Empty res_list for {pdbid}")


    res_list_pdb_dir = os.path.join(
                                os.path.dirname(complex_path),
                                f'Pocket_clean_{pocket_dis}A.pdb'
                                )
    if not os.path.exists(res_list_pdb_dir):
        res2pdb(res_list, pdbid, res_list_pdb_dir)
    res_list_coord = [res for res in parser.get_structure(pdbid, res_list_pdb_dir).get_residues() if res in res_list ]
    pos_p,_ = get_pro_coord(res_list_coord)  
    

    output_dict, _, _, _, _ = parse_PDB(
        res_list_pdb_dir,
        parse_all_atoms = True,
        parse_atoms_with_zero_occupancy = True,
    )
    features_dict = featurize(
                        output_dict
    ) 
    features_dict["S"] = features_dict["S"].to(torch.int64)

    # get backbone dihedrals as features
    # BB_atoms = features_dict["xyz_37"][0][:,:3, :]
    # BB_atoms =torch.cat([BB_atoms, torch.zeros(BB_atoms.shape[0],11, BB_atoms.shape[-1])], dim=1)  # [14, 3]
    # atoms_O = features_dict["xyz_37"][0][:, 4, :]
    # BB_atoms[:, 3, :] = atoms_O
    # BB_D,_ = calc_bb_dihedrals(BB_atoms)
    full_pdb_path = os.path.join(complex_path.split('v2020-other-PL')[0],
                                  "protein_remove_extra_chains_10A",
                                    f"{pdbid}_protein.pdb")
    # BB_D = Pocket_BB_D(full_pdb_path, output_dict["chain_letters"], 
    #                    output_dict["R_idx"])
    # print(BB_D)
    Checkpoint_path_ProteinMPNN = "../LigandMPNN/model_params/proteinmpnn_v_48_020.pt"
    ProteinMPNN_feat = get_protein_mpnn_encoder_features(full_pdb_path,
                                                          output_dict["chain_letters"],
                                                          output_dict["R_idx"],
                                                          Checkpoint_path_ProteinMPNN,
                                                            device='cpu')
                                                          
    

    chain_mask = torch.ones(features_dict["mask"].shape[0], dtype=torch.int64)
    features_dict["chain_mask"] = chain_mask.unsqueeze(0)
    torsion_dict = make_torsion_features(
                        features_dict,
                        repack_everything= False
                      
    )
    if len(pos_p) != len(torsion_dict["aatype"][0]):
        raise ValueError(f"Length of pos_p {len(pos_p)} and aatype {len(torsion_dict['aatype'][0])} not equal in {pdbid}")


    esm_feat = torch.load(os.path.join(os.path.dirname(complex_path), f"Pocket_clean_{pocket_dis}A_esm3.pt"))
    
    data_p = Data(
                x=torsion_dict["all_atom_positions"][0], # coordinates of atoms [N, 37,3]
                x_mask = features_dict["mask"][0], # mask of atoms [N]
                seq=features_dict["S"][0], # residue type [N]
                tor_an_gt=torsion_dict["torsions_true"][0], # [N,4,2] 
                aa_type=torsion_dict["aatype"][0],  
                chain_label = features_dict["chain_labels"][0], # 
                R_idx = features_dict["R_idx"][0],
                pos_p = pos_p,
                # BB_D = BB_D,
                ProteinMPNN_feat = ProteinMPNN_feat,
                esm_feat = esm_feat

                )
    data_p.__setitem__('c_size', torch.LongTensor([data_p.x.shape[0]]))

    torch.save(data_l, save_path_l)
    torch.save(data_p, save_path_aa)


class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data,  collate_fn=data.collate_fn, **kwargs)#

class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, data_df, dis_threshold=10, graph_type='MPNN', ref_pocket=False, num_process=1, create=False):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.graph_type = graph_type
        self.create = create
        self.graph_paths = None
        self.complex_ids = None
        self.num_process = num_process
        self.ref_pocket = ref_pocket
        self._pre_process()

    def _pre_process(self):
        data_dir = self.data_dir
        data_df = self.data_df
        graph_type = self.graph_type
        dis_thresholds = repeat(self.dis_threshold, len(data_df))
        ref_pockets = repeat(self.ref_pocket, len(data_df))
        # pKa_list = []
        graph_path_l_list = []
        graph_path_aa_list = []
        complex_path_list = []
        complex_id_list = []
        for i, row in data_df.iterrows():
            cid = row['pdb']
            if type(cid) != str:
                cid = str(int(cid))
            complex_dir = os.path.join(data_dir, "v2020-other-PL",cid)#  
            graph_path_l = os.path.join(complex_dir, f"{graph_type}-{cid}_l_{self.dis_threshold}A.pyg")
            graph_path_aa = os.path.join(complex_dir, f"{graph_type}-{cid}_aa_{self.dis_threshold}A.pyg")
            complex_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.rdkit")

            complex_path_list.append(complex_path)
            complex_id_list.append(cid)
            # pKa_list.append(pKa)
            graph_path_l_list.append(graph_path_l)
            graph_path_aa_list.append(graph_path_aa)

        if self.create:
            print('Generate complex graph...')
            #multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(mols2graphs,
                            zip(complex_path_list, complex_id_list, graph_path_l_list,graph_path_aa_list, ref_pockets, dis_thresholds))
            pool.close()
            pool.join()

        self.graph_paths_l = graph_path_l_list
        self.graph_paths_aa = graph_path_aa_list
        self.complex_ids = complex_id_list

    def __getitem__(self, idx):
        
        
        return torch.load(self.graph_paths_l[idx]),  torch.load(self.graph_paths_aa[idx])
            
    

    def collate_fn(self, data_list):
        batchA = Batch.from_data_list([data[0] for data in data_list])
        # print([data[1] for data in data_list][0])
        # print(data_list)
        batchB = Batch.from_data_list([data[1] for data in data_list])



        
        # lig_scope = []
        # amino_acid_scope = []
        # start_atom = 0
        # start_amino_acid = 0

        # for i in range(len(batchA)):
        #     graphA = batchA[i]
        #     graphB = batchB[i]
        #     # print(data_list[0][2])
        #     atom_count_A = graphA.num_nodes
        #     atom_count_B = graphB.num_nodes
                
        #     lig_scope.append((start_atom, atom_count_A))
        #     amino_acid_scope.append((start_amino_acid, atom_count_B))
        #     start_atom += atom_count_A
        #     start_amino_acid += atom_count_B
            
        batch = {'ligand_features': batchA,
                'protein_features': batchB,
                }

        return batch

    def __len__(self):
        return len(self.data_df)

if __name__ == '__main__':
    # pdbid = "1a0q"
    # complex_path = f'../PLmodel/supervised/data/pdbbind/v2020-other-PL/{pdbid}/{pdbid}_10A.rdkit'

    # save_path_l = os.path.dirname(complex_path) + f'/{pdbid}_ligand.pkl'
    # save_path_aa = os.path.dirname(complex_path) + f'/{pdbid}_aa.pkl'
    # mols2graphs(complex_path,pdbid,save_path_l,save_path_aa,ref_pocket=False,pocket_dis=10)

    data_root = '../PLmodel/supervised/data/pdbbind'
    
    data_dir = os.path.join(data_root, 'pdbbind')
    data_df = pd.read_csv(os.path.join(data_root, 'data.csv'))
    
    # # three hours
    toy_set = GraphDataset(data_root, data_df, graph_type='MPNN', dis_threshold=10, create=True)
    print('finish!')

   
    


