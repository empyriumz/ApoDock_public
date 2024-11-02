import os
import pickle
import torch
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit import rdBase
from tqdm import tqdm
import glob
import torch
import torch.nn.functional as F
from io import StringIO
import sys
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBIO import Select
import scipy
import scipy.spatial
from rdkit.Geometry import Point3D
from common import residue_constants, r3
import gvp
import gvp.data
def read_mol(sdf_fileName, mol2_fileName, verbose=False):
    rdBase.LogToPythonStderr()
    stderr = sys.stderr
    sio = sys.stderr = StringIO()
    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)
    problem = False
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        sm = Chem.MolToSmiles(mol)
    except Exception as e:
        sm = str(e)
        problem = True
    if problem:
        mol = Chem.MolFromMol2File(mol2_fileName, sanitize=False)
        problem = False
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            sm = Chem.MolToSmiles(mol)
            problem = False
        except Exception as e:
            sm = str(e)
            problem = True

    if verbose:
        print(sio.getvalue())
    sys.stderr = stderr
    return mol, problem

def l2_normalize(v, dim=-1, epsilon=1e-12):
    norms = torch.sqrt(torch.sum(torch.square(v), dim=dim, keepdims=True) + epsilon)
    return v / norms

def batched_select(params, indices, dim=None, batch_dims=0):
    params_shape, indices_shape = list(params.shape), list(indices.shape)
    assert params_shape[:batch_dims] == indices_shape[:batch_dims]
   
    def _permute(dim, dim1, dim2):
        permute = []
        for i in range(dim):
            if i == dim1:
                permute.append(dim2)
            elif i == dim2:
                permute.append(dim1)
            else:
                permute.append(i)
        return permute

    if dim is not None and dim != batch_dims:
        params_permute = _permute(len(params_shape), dim1=batch_dims, dim2=dim)
        indices_permute = _permute(len(indices_shape), dim1=batch_dims, dim2=dim)
        prams = torch.permute(params, params_permute)
        indices = torch.permute(indices, params_permute)
        params_shape, indices_shape = list(params.shape), list(indices.shape)

    params, indices = torch.reshape(params, params_shape[:batch_dims+1] + [-1]), torch.reshape(indices, list(indices_shape[:batch_dims]) + [-1, 1])

    # indices = torch.tile(indices, params.shape[-1:])
    indices = indices.repeat([1] * (params.ndim - 1) + [params.shape[-1]])

    batch_params = torch.gather(params, batch_dims, indices.to(dtype=torch.int64))

    output_shape = params_shape[:batch_dims] + indices_shape[batch_dims:] + params_shape[batch_dims+1:]

    if dim is not None and dim != batch_dims:
        prams = torch.permute(params, params_permute)
        indices = torch.permute(indices, params_permute)
        
    return torch.reshape(batch_params, output_shape)

def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]

def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in rc.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in  residue_constants.restypes:
        residue_name =  residue_constants.restype_1to3[residue_name]
        residue_chi_angles =  residue_constants.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([ residue_constants.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append(
                [0, 0, 0, 0]
            )  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return chi_atom_indices



def atom37_to_torsion_angles(aatype, all_atom_pos, all_atom_mask):
    num_batch, num_res = aatype.shape
    device = aatype.device

    prev_all_atom_pos = F.pad(all_atom_pos[:, :-1], [0, 0, 0, 0, 1, 0])
    prev_all_atom_mask = F.pad(all_atom_mask[:, :-1], [0, 0, 1, 0])

    # shape (B, N, atoms=4, xyz=3)
    pre_omega_atom_pos = torch.cat([
        prev_all_atom_pos[:, :, 1:3, :],  # prev CA, C
        all_atom_pos[:, :, 0:2,:]  # this N, CA
        ], dim=-2)
    phi_atom_pos = torch.cat([
        prev_all_atom_pos[:, :, 2:3, :],  # prev C
        all_atom_pos[:, :, 0:3, :]  # this N, CA, C
        ], dim=-2)
    psi_atom_pos = torch.cat([
        all_atom_pos[:, :, 0:3, :],  # this N, CA, C
        all_atom_pos[:, :, 4:5, :]  # this O
        ], dim=-2)

    # Shape [batch, num_res]
    pre_omega_mask = torch.logical_and(
            torch.all(prev_all_atom_mask[:, :, 1:3], dim=-1),  # prev CA, C
            torch.all(all_atom_mask[:, :, 0:2], dim=-1))  # this N, CA
    phi_mask = torch.logical_and(
            prev_all_atom_mask[:, :, 2], # prev C
            torch.all(all_atom_mask[:, :, 0:3], dim=-1))  # this N, CA, C
    psi_mask = torch.logical_and(
        torch.all(all_atom_mask[:, :, 0:3], dim=-1),# this N, CA, C
        all_atom_mask[:, :, 4])  # this O


    # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
    # Select atoms to compute chis. Shape: [batch, num_res, chis=4, atoms=4].
    # print(residue_constants.chi_angles_atom_indices)
    atom_indices = batched_select(torch.tensor(residue_constants.chi_angles_atom_indices, device=device), aatype)
    # Gather atom positions. Shape: [batch, num_res, chis=4, atoms=4, xyz=3].
    chis_atom_pos = batched_select(all_atom_pos, atom_indices, batch_dims=2)
    # print(chis_atom_pos[0,:,:][0])
    # Compute the chi angle mask. I.e. which chis angles exist according to the
    # aatype. Shape [batch, num_res, chis=4].
    chis_mask = batched_select(torch.tensor(residue_constants.chi_angles_mask, device=device), aatype)
    # print(chis_mask[0,:,:][0])
    # Gather the chi angle atoms mask. Shape: [batch, num_res, chis=4, atoms=4].
    chi_angle_atoms_mask = batched_select(all_atom_mask, atom_indices, batch_dims=2)
    # print(chi_angle_atoms_mask[0,:,:][0])
    # Check if all 4 chi angle atoms were set. Shape: [batch, num_res, chis=4].
    chi_angle_atoms_mask = torch.all(chi_angle_atoms_mask, dim=-1)
    chis_mask = torch.logical_and(chis_mask, chi_angle_atoms_mask)

    # Shape (B, N, torsions=7, atoms=4, xyz=3)
    torsions_atom_pos = torch.cat([
        pre_omega_atom_pos[:, :, None, :, :],
        phi_atom_pos[:, :, None, :, :],
        psi_atom_pos[:, :, None, :, :],
        chis_atom_pos], dim=2)
    
    # shape (B, N, torsions=7)
    torsion_angles_mask = torch.cat(
        [pre_omega_mask[:, :, None],
         phi_mask[:, :, None],
         psi_mask[:, :, None],
         chis_mask
        ], dim=2)

    # r3.Rigids (B, N, torsions=7)
    torsion_frames = r3.rigids_from_3_points(
            torsions_atom_pos[:, :, :, 1, :],
            torsions_atom_pos[:, :, :, 2, :],
            torsions_atom_pos[:, :, :, 0, :])

    # r3.Vecs (B, N, torsions=7)
    forth_atom_rel_pos = r3.rigids_mul_vecs(
        r3.invert_rigids(torsion_frames),
        torsions_atom_pos[:, :, :, 3, :])

    # np.ndarray (B, N, torsions=7, sincos=2)
    torsion_angles_sin_cos = torch.stack(
        [forth_atom_rel_pos[...,2], forth_atom_rel_pos[...,1]], dim=-1)
    torsion_angles_sin_cos = torsion_angles_sin_cos / torch.sqrt(
        torch.sum(torch.square(torsion_angles_sin_cos), dim=-1, keepdims=True)
        + 1e-8)
    
    chi_is_ambiguous = batched_select(
        torch.tensor(residue_constants.chi_pi_periodic, device=device), aatype)
    mirror_torsion_angles = torch.cat(
        [torch.ones([num_batch, num_res, 3], device=device),
         1.0 - 2.0 * chi_is_ambiguous], dim=-1)
    alt_torsion_angles_sin_cos = (
        torsion_angles_sin_cos * mirror_torsion_angles[:, :, :, None])
    
    return {
        'torsion_angles_sin_cos': torsion_angles_sin_cos,  # (B, N, 7, 2)
        'alt_torsion_angles_sin_cos': alt_torsion_angles_sin_cos,  # (B, N, 7, 2)
        'torsion_angles_mask': torsion_angles_mask  # (B, N, 7)
    }    

def masked_mean(mask, value, dim, eps=1e-4):
    mask = mask.expand(*value.shape)
    return torch.sum(mask * value, dim=dim) / (eps + torch.sum(mask, dim=dim))

def supervised_chi_loss(
    angles_sin_cos: torch.Tensor,
    unnormalized_angles_sin_cos: torch.Tensor,
    aatype: torch.Tensor,
    seq_mask: torch.Tensor,
    chi_mask: torch.Tensor,
    chi_angles_sin_cos: torch.Tensor,
    chi_weight: float,
    angle_norm_weight: float,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    """
        Implements Algorithm 27 (torsionAngleLoss)

        Args:
            angles_sin_cos:
                [*, N, 7, 2] predicted angles
            unnormalized_angles_sin_cos:
                 [*, N, 4, 2] predicted angles, but unnormalized
            aatype:
                [*, N] residue indices
            seq_mask:
                [*, N] sequence mask
            chi_mask:
                [*, N, 4] angle mask
            chi_angles_sin_cos:
                [*, N, 4, 2] ground truth angles
            chi_weight:
                Weight for the angle component of the loss
            angle_norm_weight:
                Weight for the normalization component of the loss
        Returns:
            [*] loss tensor
    """

    pred_angles = angles_sin_cos[..., 3:, :]
    residue_type_one_hot = torch.nn.functional.one_hot(
        aatype,
        residue_constants.restype_num + 1,
    )
    chi_pi_periodic = torch.einsum(
        "...ij,jk->...ik",
        residue_type_one_hot.type(angles_sin_cos.dtype),
        angles_sin_cos.new_tensor(residue_constants.chi_pi_periodic),
    )
    # print("chi_pi_periodic",chi_pi_periodic.shape)
    true_chi = chi_angles_sin_cos[None]
    # print("ture_chi", true_chi.shape)
    shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1)
    # print("shifted_mask",shifted_mask.shape)
    true_chi_shifted = shifted_mask * true_chi
    sq_chi_error = torch.sum((true_chi - pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum(
        (true_chi_shifted - pred_angles) ** 2, dim=-1
    )
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)

    # The ol' switcheroo
    sq_chi_error = sq_chi_error.permute(
        *range(len(sq_chi_error.shape))[1:-2], 0, -2, -1
    )

    sq_chi_loss = masked_mean(
        chi_mask[..., None, :, :], sq_chi_error, dim=(-1, -2, -3)
    )

    loss = chi_weight * sq_chi_loss

    angle_norm = torch.sqrt(
        torch.sum(unnormalized_angles_sin_cos[None] ** 2, dim=-1) + eps
    )
    norm_error = torch.abs(angle_norm - 1.0)
    norm_error = norm_error.permute(
        *range(len(norm_error.shape))[1:-2], 0, -2, -1
    )
    angle_norm_loss = masked_mean(
        seq_mask[..., None, :, None], norm_error, dim=(-1, -2, -3)
    )

    loss = loss + angle_norm_weight * angle_norm_loss

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss

def pos_to_atom37(batch_pos, atom_names):
    num_batch, num_res, num_atom, _ = batch_pos.shape
    atom37 = torch.zeros([num_batch, num_res, 37, 3], device=batch_pos.device, dtype=batch_pos.dtype)
    atom37_mask = torch.zeros([num_batch, num_res, 37], device=batch_pos.device, dtype=torch.bool)

    # 确定有效位置（非NaN, 非0）
    valid_pos = ~torch.isnan(batch_pos).any(dim=-1) & (batch_pos != 0).any(dim=-1)

    # 有效的atom_index，不等于37
    valid_indices = (atom_names != 37)

    # 综合条件检查，选择有效的原子索引
    valid = valid_pos & valid_indices

    # 使用gather和scatter更新atom37和atom37_mask
    for i in range(37):  # 仅有37种可能的位置
        index_mask = (atom_names == i) & valid  # 获取当前索引的有效位置掩码
        expanded_mask = index_mask.unsqueeze(-1).expand(-1, -1, -1, 3)  # 扩展掩码到坐标维度
        selected_pos = torch.where(expanded_mask, batch_pos, torch.zeros_like(batch_pos))
        atom37[:, :, i, :] = selected_pos.sum(dim=2)  # 将选中的位置求和（因为只有一个有效值，求和等同于选取）
        atom37_mask[:, :, i] = index_mask.any(dim=2)  # 任何有效的位置都应更新掩码

    return atom37, atom37_mask

   
  

 

  

  
    





def write_renumbered_sdf(toFile, sdf_fileName, mol2_fileName):
    # read in mol
    mol, _ = read_mol(sdf_fileName, mol2_fileName)
    # reorder the mol atom number as in smiles.
    m_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
    mol = Chem.RenumberAtoms(mol, m_order)
    w = Chem.SDWriter(toFile)
    w.write(mol)
    w.close()
def read_pdbbind_data(fileName):
    with open(fileName) as f:
        a = f.readlines()
    info = []
    for line in a:
        if line[0] == '#':
            continue
        lines, ligand = line.split('//')
        pdb, resolution, year, affinity, raw = lines.strip().split('  ')
        ligand = ligand.strip().split('(')[1].split(')')[0]
        # print(lines, ligand)
        info.append([pdb, resolution, year, affinity, raw, ligand])
    info = pd.DataFrame(info, columns=['pdb', 'resolution', 'year', 'affinity', 'raw', 'ligand'])
    info.year = info.year.astype(int)
    info.affinity = info.affinity.astype(float)
    return info
three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

def get_protein_feature(res_list):
    # protein feature extraction code from https://github.com/drorlab/gvp-pytorch
    # ensure all res contains N, CA, C and O
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    # construct the input for ProteinGraphDataset
    # which requires name, seq, and a list of shape N * 4 * 3
    structure = {}
    structure['name'] = "placeholder"
    structure['seq'] = "".join([three_to_one.get(res.resname) for res in res_list])
    coords = []
    ca= []
    # print('get_protein_feature!')
    for res in res_list:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]:
            if atom == res['CA']:
                ca.append(list(atom.coord))
            res_coords.append(list(atom.coord))
        coords.append(res_coords)
    structure['coords'] = coords
    torch.set_num_threads(1)        # this reduce the overhead, and speed up the process for me.
    dataset = gvp.data.ProteinGraphDataset([structure])
    protein = dataset[0]
    return protein.x, protein.seq, protein.node_s, protein.node_v, protein.edge_index, protein.edge_s, protein.edge_v

def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list

def remove_hetero_and_extract_ligand(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    # get all regular protein residues. and ligand.
    clean_res_list = []
    ligand_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if (not ensure_ca_exist) or ('CA' in res):
                # in rare case, CA is not exists.
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        elif hetero == 'W':
            # is water, skipped.
            continue
        else:
            ligand_list.append(res)
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list, ligand_list

def get_res_unique_id(residue):
    pdb, _, chain, (_, resid, insertion) = residue.full_id
    unique_id = f"{chain}_{resid}_{insertion}"
    return unique_id

def save_cleaned_protein(c, proteinFile):
    res_list = list(c.get_residues())
    clean_res_list, ligand_list = remove_hetero_and_extract_ligand(res_list)
    res_id_list = set([get_res_unique_id(residue) for residue in clean_res_list])

    io=PDBIO()
    class MySelect(Select):
        def accept_residue(self, residue, res_id_list=res_id_list):
            if get_res_unique_id(residue) in res_id_list:
                return True
            else:
                return False
    io.set_structure(c)
    io.save(proteinFile, MySelect())
    return clean_res_list, ligand_list

def split_protein_and_ligand(c, pdb, ligand_seq_id, proteinFile, ligandFile):
    clean_res_list, ligand_list = save_cleaned_protein(c, proteinFile)
    chain = c.id
    # should take a look of this ligand_list to ensure we choose the right ligand.
    seq_id = ligand_seq_id
    # download the ligand in sdf format from rcsb.org. because we pdb format doesn't contain bond information.
    # you could also use openbabel to do this.
    url = f"https://models.rcsb.org/v1/{pdb}/ligand?auth_asym_id={chain}&auth_seq_id={seq_id}&encoding=sdf&filename=ligand.sdf"
    r = requests.get(url)
    open(ligandFile , 'wb').write(r.content)
    return clean_res_list, ligand_list




def generate_sdf_from_smiles_using_rdkit(smiles, rdkitMolFile, shift_dis=30, fast_generation=False):
    mol_from_rdkit = Chem.MolFromSmiles(smiles)
    if fast_generation:
        # conformation generated using Compute2DCoords is very fast, but less accurate.
        mol_from_rdkit.Compute2DCoords()
    else:
        mol_from_rdkit = generate_conformation(mol_from_rdkit)
    coords = mol_from_rdkit.GetConformer().GetPositions()
    new_coords = coords + np.array([shift_dis, shift_dis, shift_dis])
    write_with_new_coords(mol_from_rdkit, new_coords, rdkitMolFile)


    # save protein chains that belong to chains_in_contact
    class MySelect(Select):
        def accept_residue(self, residue, chains_in_contact=chains_in_contact):
            pdb, _, chain, (_, resid, insertion) = residue.full_id
            if chain in chains_in_contact:
                return True
            else:
                return False

    io=PDBIO()
    io.set_structure(s)
    io.save(toFile, MySelect())

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def create_dir(dir_list):
    assert  isinstance(dir_list, list) == True
    for d in dir_list:
        if not os.path.exists(d):
            os.makedirs(d)

def save_model_dict(model, model_dir, msg):
    model_path = os.path.join(model_dir, msg + '.pt')
    torch.save(model.state_dict(), model_path)
    print("model has been saved to %s." % (model_path))

def load_model_dict(model, ckpt):
    model.load_state_dict(torch.load(ckpt))

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)  
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

def write_pickle(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def read_pickle(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj



class BestMeter(object):
    """Computes and stores the best value"""

    def __init__(self, best_type):
        self.best_type = best_type  
        self.count = 0      
        self.reset()

    def reset(self):
        if self.best_type == 'min':
            self.best = float('inf')
        else:
            self.best = -float('inf')

    def update(self, best):
        self.best = best
        self.count = 0

    def get_best(self):
        return self.best

    def counter(self):
        self.count += 1
        return self.count


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)

        return self.avg
