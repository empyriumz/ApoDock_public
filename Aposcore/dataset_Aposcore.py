# %%
import os
import pandas as pd
import numpy as np

import pickle
from scipy.spatial import distance_matrix
import multiprocessing
from itertools import repeat
import networkx as nx
import torch 
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit import RDLogger
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from utils import *
from common.residue_constants import atom_order
import warnings
from threading import Lock
my_lock = Lock()
RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

# %%
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

    return x, edge_index, edge_features


def inter_graph(ligand, pocket, dis_threshold=5.):
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    graph_inter = nx.Graph()
    pos_l = ligand.GetConformer().GetPositions()
    pos_p = pocket.GetConformer().GetPositions()

    # 添加配体-配体和口袋-口袋之间的边
    dis_matrix_l = distance_matrix(pos_l, pos_l)
    dis_matrix_p = distance_matrix(pos_p, pos_p)
    dis_matrix_lp = distance_matrix(pos_l, pos_p)

    node_idx_l = np.where(dis_matrix_l < dis_threshold)
    for i, j in zip(node_idx_l[0], node_idx_l[1]):
        graph_inter.add_edge(i, j, feats=torch.tensor([1, 0, 0, dis_matrix_l[i, j]]))

    node_idx_p = np.where(dis_matrix_p < dis_threshold)
    for i, j in zip(node_idx_p[0], node_idx_p[1]):
        graph_inter.add_edge(i + atom_num_l, j + atom_num_l, feats=torch.tensor([0, 1, 0, dis_matrix_p[i, j]]))

    node_idx_lp = np.where(dis_matrix_lp < dis_threshold)
    for i, j in zip(node_idx_lp[0], node_idx_lp[1]):
        graph_inter.add_edge(i, j + atom_num_l, feats=torch.tensor([0, 0, 1, dis_matrix_lp[i, j]]))

    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v, _ in graph_inter.edges(data=True)]).T
    edge_attrs_inter = torch.stack([feats['feats'] for _, _, feats in graph_inter.edges(data=True)]).float()

    return edge_index_inter, edge_attrs_inter

def get_protein_res_class(protein, pocket):
    pro_res_list = get_clean_res_list(protein.get_residues(), verbose=False, ensure_ca_exist=True)
    # print(pro_res_list)
    pocket_res_list = get_clean_res_list(pocket.get_residues(), verbose=False, ensure_ca_exist=True)
    # print(pocket_res_list)
    x_aa, seq, node_s, node_v, edge_index, edge_s, edge_v = get_protein_feature(pro_res_list)
    class_list = []
    for res in pro_res_list:
        if res in pocket_res_list:
            class_list.append(1)
        else:
            class_list.append(0)
    # class_list = torch.tensor(class_list).float
    class_list = torch.FloatTensor(class_list)
    if len(class_list) != len(seq):
        raise ValueError(f'The length of class_list is not equal to the length of seq of protein {protein.get_id()}')
    return class_list,x_aa, seq, node_s, node_v, edge_index, edge_s, edge_v

def get_CA_coord(res):
    # 获取残基的CA原子的坐标
    for atom in [res['N'], res['CA'], res['C'], res['O']]:
            if atom == res['CA']:
                return atom.coord
            


def get_pro_coord(pro,max_atom_num=24):
    # 获取残基的所有原子的坐标，包括side chain
    coords_pro = []
    coords_main_pro = [] # 只包含主链N,CA,C原子的坐标
    atom_names = [] # 所以原子的名字
    for res in pro:
        # print(len(res.get_atoms()))
        # break
        atoms = res.get_atoms()
        # print(len(atoms))
        # print(dir(atoms))
        # break
        coords_res = []
        coords_res_main = []
        atom_names_res = []
        for atom in atoms:
            # print(atom.name)
            # break
            if atom.name in ['N', 'CA', 'C']:
                coords_res_main.append(atom.coord)
            coords_res.append(atom.coord)
            atom_names_res.append(atom.name)
            
        coords_res_main = np.array(coords_res_main)
        coords_res = np.array(coords_res)

        # print(len(coords_res))
        coords_res = np.concatenate([coords_res, np.full((max_atom_num - len(coords_res), 3),np.nan)], axis=0)
        # 将atom_name_res补全到max_atom_num
        # atom_names_res = atom_names_res + ['nan'] * (max_atom_num - len(atom_names_res))
        # atom_names_res = np.array(atom_names_res)
        # 根据索引将atom_name_res 转换为index
        atom_names_res = [atom_order[atom] for atom in atom_names_res]
        # 加上逗号
        atom_names_res = np.concatenate([atom_names_res, np.full((max_atom_num - len(atom_names_res),),37)], axis=0)
        # print(atom_names_res)
        # print(atom_names_res.shape)
        # print(coords.shape)
        # break
        coords_main_pro.append(coords_res_main)
        coords_pro.append(coords_res) 
        atom_names.append(atom_names_res)
    # print(coords_pro[0].shape)
        
    coords_pro = torch.tensor(coords_pro)
    coords_main_pro = torch.tensor(coords_main_pro)
    atom_names = torch.tensor(atom_names, dtype=torch.long)
    # print(coords_pro.shape)
    # print(coords_main_pro.shape)
    # print(atom_names[0])
    return coords_pro, coords_main_pro, atom_names
    



# %%
def mols2graphs(complex_path, pdbid, label, save_path_l,save_path_aa, ref_pocket=False, pocket_dis = 5):
    print(pdbid)
    with open(complex_path, 'rb') as f:
        ligand, pocket = pickle.load(f)

    atom_num_l = ligand.GetNumAtoms()

    x_l, edge_index_l,edge_features_l, degrees = mol2graph(ligand)

    pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
    # print('pos_l:', pos_l.shape)    
    parser = PDBParser(QUIET=True)
    if ref_pocket:
        pocket_id = pdbid.split('_')[0]
        pocket_pdb = f"./data/pdbbind/v2020-other-PL/{pocket_id}/Pocket_{pocket_dis}A.pdb"
    else:

        pocket_pdb = os.path.join(
                                os.path.dirname(complex_path),
                                f'Pocket_{pocket_dis}A.pdb'
                                )

        
    # protein_pdb = os.path.join(
    #                         os.path.dirname(complex_path).split('v2020-other-PL')[0],
    #                         'protein_remove_extra_chains_10A',
    #                         f'{pdbid}_protein.pdb'
    #                         )
    
    # class_aa,x_aa, seq, node_s, \
    #       node_v, edge_index, edge_s, edge_v \
    #             = get_protein_res_class(
    #                         parser.get_structure(pdbid, protein_pdb),
    #                         parser.get_structure(pdbid, pocket_pdb)
    #                         )
    res_list_feature = get_clean_res_list(parser.get_structure(pdbid, pocket_pdb).get_residues(), verbose=False, ensure_ca_exist=True)
    x_aa, seq, node_s, \
        node_v, edge_index, edge_s, edge_v \
                  = get_protein_feature(
                                res_list_feature
                                )
    pos_p_ca = torch.FloatTensor([get_CA_coord(res) for res in res_list_feature])

    res_list_coord = [res for res in parser.get_structure(pdbid, pocket_pdb).get_residues() if res in res_list_feature ]
    # print(len(res_list_coord))
    # print(len(res_list_feature))
    if len(res_list_coord) != len(res_list_feature):
        raise ValueError(f'The length of res_list_coord({len(res_list_coord)}) is not equal to the length of res_list_feature({len(res_list_feature)}) in {pdbid}')

    pos_p,pos_main_p, atom_names = get_pro_coord(res_list_coord)

                              


    c_size_l = atom_num_l
    c_size_aa = len(seq)
    y = torch.FloatTensor([label])
    data_l = Data(
                x=x_l,
                edge_index=edge_index_l,
                edge_attr=edge_features_l,
                pos = pos_l,
                y=y,
                degrees = degrees
                )
    data_l.__setitem__('c_size', torch.LongTensor([c_size_l]))
    data_aa = Data(x_aa=x_aa,
                seq=seq,
                node_s=node_s,
                node_v=node_v,
                edge_index=edge_index,
                edge_s=edge_s,
                edge_v=edge_v,
                y=y,
                pos = pos_p,
                pos_main_p = pos_main_p,
                atom_names = atom_names
                )
    data_aa.__setitem__('c_size', torch.LongTensor([c_size_aa]))

    torch.save(data_l, save_path_l)
    torch.save(data_aa, save_path_aa)

# %%
class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data,  collate_fn=data.collate_fn, **kwargs)#

class GraphDataset(Dataset):
    """
    This class is used for generating graph objects using multi process
    """
    def __init__(self, data_dir, data_df, dis_threshold=8, graph_type='ConBAP', ref_pocket=False, num_process=28, create=False):
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
        pKa_list = []
        graph_path_l_list = []
        graph_path_aa_list = []
        complex_path_list = []
        complex_id_list = []
        for i, row in data_df.iterrows():
            cid, pKa = row['pdb'], float(row['affinity'])
            if type(cid) != str:
                cid = str(int(cid))
            complex_dir = os.path.join(data_dir, cid)#  "v2020-other-PL",
            graph_path_l = os.path.join(complex_dir, f"{graph_type}-{cid}_l_{self.dis_threshold}A.pyg")
            graph_path_aa = os.path.join(complex_dir, f"{graph_type}-{cid}_aa_{self.dis_threshold}A.pyg")
            complex_path = os.path.join(complex_dir, f"{cid}_{self.dis_threshold}A.rdkit")

            complex_path_list.append(complex_path)
            complex_id_list.append(cid)
            pKa_list.append(pKa)
            graph_path_l_list.append(graph_path_l)
            graph_path_aa_list.append(graph_path_aa)

        if self.create:
            print('Generate complex graph...')
            #multi-thread processing
            pool = multiprocessing.Pool(self.num_process)
            pool.starmap(mols2graphs,
                            zip(complex_path_list, complex_id_list, pKa_list, graph_path_l_list,graph_path_aa_list, ref_pockets, dis_thresholds))
            pool.close()
            pool.join()

        self.graph_paths_l = graph_path_l_list
        self.graph_paths_aa = graph_path_aa_list
        self.complex_ids = complex_id_list

    def __getitem__(self, idx):
        
        
        return torch.load(self.graph_paths_l[idx]),  torch.load(self.graph_paths_aa[idx]) 
            
    

    def collate_fn(self, data_list):
        batchA = Batch.from_data_list([data[0] for data in data_list])
        batchB = Batch.from_data_list([data[1] for data in data_list])


        
        lig_scope = []
        amino_acid_scope = []
        start_atom = 0
        start_amino_acid = 0

        for i in range(len(batchA)):
            graphA = batchA[i]
            graphB = batchB[i]
            # print(data_list[0][2])
            atom_count_A = graphA.num_nodes
            atom_count_B = graphB.num_nodes
                
            lig_scope.append((start_atom, atom_count_A))
            amino_acid_scope.append((start_amino_acid, atom_count_B))
            start_atom += atom_count_A
            start_amino_acid += atom_count_B
            
        batch = {'ligand_features': batchA, 'protein_features': batchB, 'ligand_scope': lig_scope, 'protein_scope': amino_acid_scope}

        return batch

    def __len__(self):
        return len(self.data_df)

if __name__ == '__main__':
    data_root = './data/pdbbind'
    
    data_dir = os.path.join(data_root, 'pdbbind')
    data_df = pd.read_csv(os.path.join(data_root, 'data.csv'))
    
    # # three hours
    toy_set = GraphDataset(data_root, data_df, graph_type='ConBAP', dis_threshold=10, create=True)
    print('finish!')
    # data_root = "./data/docking_no_ref_pocket"
    # pdbids = os.listdir(data_root)
    # # pdbids =["1a30"]
    # # pdbids =["5c2h"]
    # for pdbid in pdbids:
    #     data_dir = os.path.join(data_root, pdbid)
    #     data_df = pd.read_csv(os.path.join(data_dir, f"{pdbid}.csv"))
    #     toy_set = GraphDataset(data_dir, data_df, graph_type='ConBAP', ref_pocket=True, dis_threshold=10, create=True)
    
    # # print('finish!')
    # parser = PDBParser(QUIET=True)
    # pocket = "./data/pdbbind/v2020-other-PL/1a1c/Pocket_8A.pdb"
    # # with open("./data/pdbbind/v2020-other-PL/1a30/1a30_8A.rdkit", 'rb') as f:
    # #     ligand, _ = pickle.load(f)
    # # pocket = parser.get_structure('1a30', pocket)
    # # res_list_feature = get_clean_res_list(parser.get_structure('1a1c', pocket).get_residues(), verbose=False, ensure_ca_exist=True)
    # res_list_coord = [res for res in parser.get_structure('1a1c', pocket).get_residues()]
    # # x_aa, seq, node_s, \
    # # node_v, edge_index, edge_s, edge_v \
    # #             = get_protein_feature(
    # #                         res_list_feature
    # #                         )
    # # print(seq)
    # # # interact_matrix = get_interaction_matrix(ligand, pocket.get_residues())
    # # # print(interact_matrix)
    # get_pro_coord(res_list_coord)

    


# %%
