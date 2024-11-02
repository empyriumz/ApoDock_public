import os
import numpy as np
from itertools import repeat
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
import sys
from dataset_Aposcore import mol2graph, get_pro_coord
from Bio.PDB import PDBParser
from utils import get_clean_res_list, get_protein_feature, load_model_dict
from Aposcore import Aposcore
from torch_scatter import scatter_add

def read_sdf_file(mol_file, save_mols=False):
    """
    This function reads a SDF file containing multiple molecules and returns a dict of RDKit molecule objects.
    """
    if not os.path.exists(mol_file):
        sys.exit(f"The MOL2 file {mol_file} does not exist")
    # Create a Mol2MolSupplier object
    mol_file_name = os.path.basename(mol_file).split(".")[0]
    # if save_mols:
    supplier = Chem.MultithreadedSDMolSupplier(mol_file, removeHs=True, sanitize=True)
    molecules = []
    molecules_name = []
    for i, mol in enumerate(supplier):
        if mol is not None:
            molecules_name.append(mol_file_name + "_" + str(i))
            molecules.append(mol)
    
    if len(molecules) == 0:
        if save_mols:
            print("No molecules pass the rdkit sanitization")
            return None, None

        supplier = Chem.SDMolSupplier(mol_file, removeHs=True, sanitize=False)
        for i, mol in enumerate(supplier):
            if mol is not None:
                molecules_name.append(mol_file_name + "_" + str(i))
                molecules.append(mol)
    # print(len(molecules))
    return molecules, molecules_name

def get_graph_data_l(mol):
    atom_num_l = mol.GetNumAtoms()
    x_l, edge_index_l,edge_features_l = mol2graph(mol)

    pos_l = torch.FloatTensor(mol.GetConformers()[0].GetPositions())

    c_size_l = atom_num_l

    data_l = Data(
                x=x_l,
                edge_index=edge_index_l,
                edge_attr=edge_features_l,
                pos = pos_l,
                )
    data_l.__setitem__('c_size', torch.LongTensor([c_size_l]))

    return data_l

def get_graph_data_p(pocket_path):
    pocket_id = os.path.basename(pocket_path)
    parser = PDBParser(QUIET=True)
    res_list_feature = get_clean_res_list(parser.get_structure(pocket_id, pocket_path).get_residues(),
                                        verbose=False,                                             
                                        ensure_ca_exist=True)
    x_aa, seq, node_s, \
    node_v, edge_index, edge_s, edge_v \
                = get_protein_feature(
                            res_list_feature
                            )
    
    c_size_aa = len(seq)
    res_list_coord = [res for res in parser.get_structure(pocket_id, pocket_path).get_residues() if res in res_list_feature]

    if len(res_list_coord) != len(res_list_feature):
        raise ValueError(f'The length of res_list_coord({len(res_list_coord)}) is not equal to the length of res_list_feature({len(res_list_feature)}) in {pdbid}')

    pos_p,_,_ = get_pro_coord(res_list_coord)

    data_aa = Data(x_aa=x_aa,
                seq=seq,
                node_s=node_s,
                node_v=node_v,
                edge_index=edge_index,
                edge_s=edge_s,
                edge_v=edge_v,
                pos = pos_p,
                )
    data_aa.__setitem__('c_size', torch.LongTensor([c_size_aa]))

    return data_aa


def mol2graphs_dock(mol_list, pocket):
    """
    mol_list dict contains rdkit mols, pocket, pocket information
    
    """
    graph_data_list_l =[]
    graph_data_name_l =[]
    for i in mol_list:
        mol = i
        graph_mol = get_graph_data_l(mol)
        graph_data_list_l.append(graph_mol)
        graph_data_name_l.append(i)

    graph_data_aa = get_graph_data_p(pocket)

    graph_data_list_aa = repeat(graph_data_aa,len(graph_data_list_l))

    return  graph_data_list_l, graph_data_name_l, graph_data_list_aa

class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data,  collate_fn=data.collate_fn, **kwargs)#

class Dataset_infer(Dataset):    

    def __init__(self, sdf_list_path, pocket_list_path):

        self.sdf_list_path = sdf_list_path
        self.pocket_list_path = pocket_list_path
        self._pre_process()

    def _pre_process(self):
        sdf_list_path = self.sdf_list_path
        pocket_list_path =  self.pocket_list_path

        graph_l_list = []
        graph_aa_list = []
        graph_name_list = []

        for mol, pocket in zip(sdf_list_path, pocket_list_path):
            mol,_ = read_sdf_file(mol, save_mols=False)
            graph_data_list_l, graph_data_name_l, graph_data_list_aa = mol2graphs_dock(mol, pocket)
            graph_l_list.extend(graph_data_list_l)
            graph_aa_list.extend(graph_data_list_aa)
            graph_name_list.extend(graph_data_name_l)

        self.graph_l_list = graph_l_list
        self.graph_aa_list = graph_aa_list
        self.graph_name_list = graph_name_list

    def __getitem__(self, idx):

        return self.graph_l_list[idx],  self.graph_aa_list[idx]
    
    def collate_fn(self, data_list):
        batchA = Batch.from_data_list([data[0] for data in data_list])
        batchB = Batch.from_data_list([data[1] for data in data_list])

        batch = {'ligand_features': batchA, 'protein_features': batchB}

        return batch

    def __len__(self):
        return len(self.graph_l_list)

def val(model, dataloader, device, dis_threshold=5.0):
    model.eval()

    probs = []
    for data in dataloader:
        
        with torch.no_grad():
            data['ligand_features'] = data['ligand_features'].to(device)
            data['protein_features'] = data['protein_features'].to(device)
            pi, sigma, mu, dist, batch = model(data)
            prob = model.calculate_probablity(pi, sigma, mu,dist)
            prob[torch.where(dist > dis_threshold)[0]] = 0.
            batch = batch.to(device)
            probx = scatter_add(prob, batch, dim=0, dim_size=batch.unique().size(0)).cpu().numpy()
            probs.append(probx)
            
    pred = np.concatenate(probs)
            
    return pred



def get_mdn_score(sdf_files, pocket_files, model, ckpt, device,dis_threshold=5.0):

    load_model_dict(model,ckpt)
    model = model.to(device)
    toy_set = Dataset_infer(sdf_files, pocket_files)
    toy_loader = PLIDataLoader(toy_set, batch_size=64, shuffle=False)

    pred = val(model, toy_loader, device, dis_threshold)
    # print(pred)  
    return pred



if __name__ == "__main__":
    # mol2_files = ["../test/posebuster_dataset/posebusters_benchmark_set/5S8I_2LY/gnina_dock_5S8I.sdf"]
    # pocket_files = ["../test/posebuster_dataset/posebusters_benchmark_set/5S8I_2LY/Pocket_10A.pdb"]
    # hid_dim = 256
    # num_heads = 4
    # dropout = 0.1
    # num_layers = 6
    # dis_threshold = 5.0
    # interact_type = 'product'
    # crossAttention = True
    # atten_active_fuc = 'softmax'
    # ckpt =f"./model/20240423_150824_ConBAP_repeat0/model/epoch-491, train_loss-0.6195, train_mdn-0.6195, train_aff--0.4271, valid_loss-0.8518, valid_mdn-0.8518, valid_aff--0.0822, coff-0.4097.pt"
    # device = torch.device('cuda:0')
    # model = bindingsite(35, 
    #                 hid_dim,
    #                 num_heads,                               
    #                 dropout,
    #                 crossAttention,
    #                 atten_active_fuc,
    #                 num_layers,
    #                 interact_type,
    #                 ).to(device)

    # pred = get_mdn_score(mol2_files, pocket_files, model, ckpt, device)

    # print(len(pred))
    mol2_file = "../test/posebuster_dataset/posebusters_benchmark_set/5SB2_1K2/gnina_dock_5SB2_1.sdf"

    read_sdf_file(mol2_file)