import torch
import numpy as np
import os
from torch_geometric.data import Data, Batch
from rdkit import Chem
from Bio.PDB import PDBParser, Superimposer
from torch_geometric.utils import to_dense_batch
from apodock.Pack_sc.resampling import resample_loop
from apodock.utils import set_random_seed
from apodock.Pack_sc.data_utils import (
    parse_PDB,
    get_clean_res_list,
    featurize,
    get_atom14_coords,
    write_full_PDB,
)
from apodock.Pack_sc.sc_utils import make_torsion_features
from apodock.Pack_sc.dataset_pack import (
    mol2graph,
    res2pdb,
    get_protein_mpnn_encoder_features,
    Pocket_BB_D,
)
from apodock.Pack_sc.openfold.data.data_transforms import make_atom14_masks
from torch.utils.data import Dataset, DataLoader
from sklearn_extra.cluster import KMedoids


def parse_pockets(pdb_files):
    """Parses protein pocket atom coordinates from PDB files."""
    parser = PDBParser(QUIET=True)
    pockets = []
    for pdb_file in pdb_files:
        structure = parser.get_structure("Pocket", pdb_file)
        atoms = [
            atom for residue in structure.get_residues() for atom in residue.get_atoms()
        ]
        pockets.append(atoms)
    return pockets


def is_symmetric_residue(residue):
    """Determines if a residue has symmetry in its structure."""
    symmetric_residues = {"PHE", "TYR", "ASP", "GLU"}
    return residue.get_resname() in symmetric_residues


def handle_symmetry(pocket1, pocket2):
    """Adjusts atom order based on symmetry for proper comparisons."""
    for i, atom1 in enumerate(pocket1):
        atom2 = pocket2[i]
        res1 = atom1.get_parent()  # Get the Residue that atom belongs to
        res2 = atom2.get_parent()  # Get the Residue that atom belongs to
        if is_symmetric_residue(res1) and is_symmetric_residue(res2):
            # Process symmetric residues, potentially adjusting atom order
            pass
    return pocket1, pocket2


def compute_rmsd_matrix(pockets):
    """Computes RMSD matrix between protein pockets, considering symmetry."""
    num_pockets = len(pockets)
    rmsd_matrix = np.zeros((num_pockets, num_pockets))
    sup = Superimposer()

    for i in range(num_pockets):
        for j in range(i + 1, num_pockets):
            assert len(pockets[i]) == len(pockets[j])
            pocket1, pocket2 = handle_symmetry(pockets[i], pockets[j])
            sup.set_atoms(pocket1, pocket2)
            rmsd_matrix[i, j] = sup.rms
            rmsd_matrix[j, i] = rmsd_matrix[i, j]

    return rmsd_matrix


def cluster_pockets(pdb_files, num_clusters=6, random_seed=42):
    """
    Clusters protein pockets based on structural similarity.

    Args:
        pdb_files: List of PDB file paths to cluster
        num_clusters: Number of clusters to create (default: 6)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        List of PDB file paths representing cluster centers
    """
    # Set random seed for KMedoids clustering
    set_random_seed(random_seed, log=False)

    pockets = parse_pockets(pdb_files)
    rmsd_matrix = compute_rmsd_matrix(pockets)
    clustering = KMedoids(
        n_clusters=num_clusters, metric="precomputed", random_state=random_seed
    ).fit(rmsd_matrix)
    cluster_centers = clustering.medoid_indices_

    cluster_names = [pdb_files[idx] for idx in cluster_centers]
    # Delete files that are not cluster centers
    for i in range(len(pdb_files)):
        if i not in cluster_centers:
            os.remove(pdb_files[i])
    return cluster_names


def read_mol(ligand):
    """Reads a molecule from various file formats."""
    input_ligand_format = ligand.split(".")[-1]
    if input_ligand_format == "sdf":
        mol = Chem.SDMolSupplier(ligand)[0]
    elif input_ligand_format == "mol":
        mol = Chem.MolFromMolFile(ligand)
    elif input_ligand_format == "mol2":
        mol = Chem.MolFromMol2File(ligand)
    elif input_ligand_format == "pdb":
        mol = Chem.MolFromPDBFile(ligand)

    if mol is None:
        raise ValueError(f"Failed to load ligand from {ligand}")
    return mol


def get_graph_data_l(mol):
    """Converts a ligand molecule to a graph representation."""
    atom_num_l = mol.GetNumAtoms()
    x_l, edge_index_l, edge_features_l, _ = mol2graph(mol)

    c_size_l = atom_num_l

    data_l = Data(
        x=x_l,
        edge_index=edge_index_l,
        edge_attr=edge_features_l,
    )
    data_l.__setitem__("c_size", torch.LongTensor([c_size_l]))

    return data_l


def get_graph_data_p(pocket_path, protein_path, ligandmpnn_path, apo2holo=False):
    """Converts a protein pocket to a graph representation."""
    pdbid = os.path.dirname(pocket_path).split("/")[-1]
    base_dir = os.path.dirname(pocket_path)
    parser = PDBParser(QUIET=True)
    res_list = get_clean_res_list(
        parser.get_structure(pdbid, pocket_path).get_residues(),
        verbose=False,
        ensure_ca_exist=True,
    )
    res_list_pdb_dir = os.path.join(base_dir, f"Pocket_clean_{pdbid}.pdb")
    if not os.path.exists(res_list_pdb_dir):
        print(f"Writing clean pocket PDB file for {pdbid}")
        res2pdb(res_list, pdbid, res_list_pdb_dir)

    output_dict, _, _, _, _ = parse_PDB(
        res_list_pdb_dir,
        parse_all_atoms=True,
        parse_atoms_with_zero_occupancy=True,
    )
    features_dict = featurize(output_dict)
    features_dict["S"] = features_dict["S"].to(torch.int64)
    chain_mask = torch.ones(features_dict["mask"].shape[0], dtype=torch.int64)
    features_dict["chain_mask"] = chain_mask.unsqueeze(0)
    torsion_dict = make_torsion_features(features_dict, repack_everything=False)
    full_pdb_path = protein_path

    BB_D = Pocket_BB_D(
        full_pdb_path, output_dict["chain_letters"], output_dict["R_idx"]
    )
    BB_D[torch.isnan(BB_D)] = 0

    Checkpoint_path_ProteinMPNN = ligandmpnn_path

    protein_mpnn_feat = get_protein_mpnn_encoder_features(
        full_pdb_path,
        output_dict["chain_letters"],
        output_dict["R_idx"],
        Checkpoint_path_ProteinMPNN,
        device="cpu",
    )

    data_p = Data(
        x=torsion_dict["all_atom_positions"][0],  # coordinates of atoms [N, 37,3]
        x_mask=features_dict["mask"][0],  # mask of atoms [N]
        seq=features_dict["S"][0],
        tor_an_gt=torsion_dict["torsions_true"][0],  # [N,4,2]
        aa_type=torsion_dict["aatype"][0],  # [N,21]
        chain_label=features_dict["chain_labels"][0],  #
        R_idx=features_dict["R_idx_original"][0],
        ProteinMPNN_feat=protein_mpnn_feat,
        BB_D=BB_D,
    )
    data_p.__setitem__("c_size", torch.LongTensor(data_p.x.shape[0]))
    return data_p


def get_complex_data(
    list_of_ligands, list_of_pockets, list_of_proteins, ligandmpnn_path, apo2holo=False
):
    """Prepares complex data for both ligands and proteins."""
    garph_data_list_l = []
    graph_data_list_p = []
    for ligand, pocket, protein in zip(
        list_of_ligands, list_of_pockets, list_of_proteins
    ):
        ligand = read_mol(ligand)
        data_l = get_graph_data_l(ligand)
        data_p = get_graph_data_p(pocket, protein, ligandmpnn_path, apo2holo)
        garph_data_list_l.append(data_l)
        graph_data_list_p.append(data_p)

    return garph_data_list_l, graph_data_list_p


def get_file_list(data_dir):
    """Gets lists of ligand and pocket files from a data directory."""
    assert len(data_dir) > 0
    pdb_files = os.listdir(data_dir)
    ligand_list = []
    pocket_list = []
    for pdbid in pdb_files:
        pocket = [
            i
            for i in os.listdir(os.path.join(data_dir, pdbid))
            if i.endswith("10A.pdb")
        ]
        assert len(pocket) > 0
        pocket = os.path.join(data_dir, pdbid, pocket[0])
        ligand = [
            i
            for i in os.listdir(os.path.join(data_dir, pdbid))
            if i.endswith("_ligand.sdf")
            or i.endswith("_ligand.mol")
            or i.endswith("_ligand.mol2")
            or i.endswith("_ligand.pdb")
            or i.endswith("_aligned.sdf")
        ]

        if len(ligand) == 0:
            ligand = os.path.join(data_dir, pdbid, f"{pdbid}.sdf")
        else:
            ligand = os.path.join(data_dir, pdbid, ligand[0])
        assert len(ligand) > 0

        ligand_list.append(ligand)
        pocket_list.append(pocket)
    return ligand_list, pocket_list


class PackDataLoader(DataLoader):
    """Custom DataLoader for protein packing data."""

    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


class Pack_infer(Dataset):
    """Dataset class for protein packing inference."""

    def __init__(
        self, ligand_list, pocket_list, protein_list, ligandmpnn_path, apo2holo=False
    ):
        self.ligand_list = ligand_list
        self.pocket_list = pocket_list
        self.protein_list = protein_list
        self.ligandmpnn_path = ligandmpnn_path
        self.apo2holo = apo2holo
        self._pre_process()

    def _pre_process(self):
        ligand_list = self.ligand_list
        pocket_list = self.pocket_list
        protein_list = self.protein_list
        ligandmpnn_path = self.ligandmpnn_path
        apo2holo = self.apo2holo
        garph_data_list_l, graph_data_list_p = get_complex_data(
            ligand_list, pocket_list, protein_list, ligandmpnn_path, apo2holo
        )
        self.graph_l_list = garph_data_list_l
        self.graph_p_list = graph_data_list_p

    def __len__(self):
        return len(self.graph_l_list)

    def __getitem__(self, idx):
        return self.graph_l_list[idx], self.graph_p_list[idx]

    def collate_fn(self, data_list):
        batchA = Batch.from_data_list([data[0] for data in data_list])
        batchB = Batch.from_data_list([data[1] for data in data_list])
        batch = {"ligand": batchA, "protein": batchB}
        return batch


def _move_batch_to_device(batch, device):
    """Helper function to move batch data to device."""
    batch["ligand"] = batch["ligand"].to(device)
    batch["protein"] = batch["protein"].to(device)
    return batch


def _process_ligand_features(model, ligand_data):
    """Process ligand features through the model's convolutional layers."""
    lig_feat = model.lin_node_lig(ligand_data.x)

    for conv in model.conv:
        lig_feat = conv(lig_feat, ligand_data.edge_index, ligand_data.edge_attr)

    return lig_feat


def _update_results_with_batch_data(results, batch, feature_dict):
    """Update results dictionary with batch data."""
    results.update(
        {
            "ligand": batch["ligand"],
            "protein": batch["protein"],
            "mask": feature_dict["mask"],
            "atom14_mask": feature_dict["atom14_mask"],
            "S": feature_dict["S"],
            "aa_type": feature_dict["aatype"],
            "R_idx": feature_dict["R_idx"],
            "chain_labels": feature_dict["chain_labels"],
            "BB_D": feature_dict["BB_D"],
        }
    )
    return results


def _perform_resampling(results, resample_args):
    """Perform resampling for each structure in the results."""
    for i in range(results["final_X"].shape[0]):
        protein = {
            "S": results["aa_type"][i],
            "X": results["final_X"][i],
            "BB_D": results["BB_D"][i],
            "X_mask": results["atom14_mask"][i],
            "residue_mask": results["mask"][i],
            "residue_index": results["R_idx"][i],
            "chi_logits": results["chi_logits"][i],
            "chi_bin_offset": results["chi_bin_offset"][i],
        }

        pred_xyz = results["final_X"][i]
        resample_xyz, _ = resample_loop(protein, pred_xyz, **resample_args)
        results["final_X"][i] = resample_xyz

    return results


def pro2feature_dict(data_aa):
    """Convert protein data to feature dictionary for the model."""
    X, X_mask, seq, tor_an_gt, chain_label, R_idx, aa_type, ProteinMPNN_feat, BB_D = (
        data_aa.x,
        data_aa.x_mask,
        data_aa.seq,
        data_aa.tor_an_gt,
        data_aa.chain_label,
        data_aa.R_idx,
        data_aa.aa_type,
        data_aa.ProteinMPNN_feat,
        data_aa.BB_D,
    )
    X, mask = to_dense_batch(X, data_aa.batch, fill_value=0)
    seq, _ = to_dense_batch(seq, data_aa.batch, fill_value=0)
    aa_type, _ = to_dense_batch(aa_type, data_aa.batch, fill_value=0)
    tor_an_gt, _ = to_dense_batch(tor_an_gt, data_aa.batch, fill_value=0)
    chain_labels, _ = to_dense_batch(chain_label, data_aa.batch, fill_value=0)
    R_idx, _ = to_dense_batch(R_idx, data_aa.batch, fill_value=0)
    protein_mpnn_feat, _ = to_dense_batch(ProteinMPNN_feat, data_aa.batch, fill_value=0)
    BB_D, _ = to_dense_batch(BB_D, data_aa.batch, fill_value=0)

    mask = mask.float()
    masks14_37 = make_atom14_masks({"aatype": aa_type})
    atom_14_mask = masks14_37["atom14_atom_exists"]
    atom_14_mask = atom_14_mask * mask[:, :, None]
    atom_37_mask = masks14_37["atom37_atom_exists"]
    atom_37_mask = atom_37_mask * mask[:, :, None]

    atom_14 = get_atom14_coords(
        X, seq, atom_14_mask, atom_37_mask, tor_an_gt, device=X.device
    )
    feature_dict = {}
    feature_dict["x"] = atom_14
    feature_dict["mask"] = mask
    feature_dict["x_37"] = X
    feature_dict["atom37_mask"] = atom_37_mask
    feature_dict["atom14_mask"] = atom_14_mask
    feature_dict["S"] = seq
    feature_dict["tor_an_gt"] = tor_an_gt
    feature_dict["chain_labels"] = chain_labels
    feature_dict["R_idx"] = R_idx
    feature_dict["aatype"] = aa_type
    feature_dict["protein_mpnn_feat"] = protein_mpnn_feat
    feature_dict["BB_D"] = BB_D

    return feature_dict


def _create_packing_dataset(
    ligand_list, pocket_list, protein_list, ligandmpnn_path, apo2holo=False
):
    """Create dataset for protein side-chain packing."""
    return Pack_infer(ligand_list, pocket_list, protein_list, ligandmpnn_path, apo2holo)


def get_letter_codes(pocket_list):
    """Get letter codes for residues from pocket files."""
    assert len(pocket_list) > 0
    CA_icodes_list = []
    Chain_letters_list = []
    for pocket in pocket_list:
        pocket_id = os.path.dirname(pocket).split("/")[-1]
        pocket = os.path.join(os.path.dirname(pocket), f"Pocket_clean_{pocket_id}.pdb")
        output_dict, _, _, CA_icodes, _ = parse_PDB(
            pocket,
            parse_all_atoms=True,
            parse_atoms_with_zero_occupancy=True,
        )
        CA_icodes_list.append(CA_icodes)
        Chain_letters_list.append(output_dict["chain_letters"])
    max_len = max([len(x) for x in CA_icodes_list])
    CA_icodes = [
        np.pad(arr, (0, max_len - len(arr)), mode="constant", constant_values="")
        for arr in CA_icodes_list
    ]
    Chain_letters = [
        np.pad(arr, (0, max_len - arr.size), mode="constant", constant_values="")
        for arr in Chain_letters_list
    ]

    return CA_icodes, Chain_letters


def _batch_input_data(
    ligand_list,
    pocket_list,
    protein_list,
    protein_filenames,
    out_dir,
    batch_size,
):
    """Group input data into batches."""
    # Create batches using full protein filenames
    filename_batches = [
        protein_filenames[i : i + batch_size]
        for i in range(0, len(protein_filenames), batch_size)
    ]

    return {
        "pdbid_batches": filename_batches,
        "out_dir": out_dir,
    }


def _batch_residue_info(ca_icodes, chain_letters, batch_size):
    """Group residue information into batches."""
    ca_icodes_batches = [
        ca_icodes[i : i + batch_size] for i in range(0, len(ca_icodes), batch_size)
    ]
    chain_letters_batches = [
        chain_letters[i : i + batch_size]
        for i in range(0, len(chain_letters), batch_size)
    ]

    return {
        "ca_icodes_batches": ca_icodes_batches,
        "chain_letters_batches": chain_letters_batches,
    }


def _clear_gpu_memory(device):
    """Clear GPU memory if using CUDA device."""
    if (
        isinstance(device, str)
        and device.startswith("cuda")
        or isinstance(device, torch.device)
        and device.type == "cuda"
    ):
        try:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Failed to clear GPU memory: {str(e)}")


def _write_pdb_files(results_list, batched_data, batched_residue_info):
    """Write PDB files for all sampled conformations."""
    packed_files_list = []

    # For each result in results_list
    for i, results in enumerate(results_list):
        for result_batch, pdbids, ca_icodes, chain_letters in zip(
            results,
            batched_data["pdbid_batches"],
            batched_residue_info["ca_icodes_batches"],
            batched_residue_info["chain_letters_batches"],
        ):
            # Validate batch consistency
            batch_len = len(result_batch["final_X"])
            if not (batch_len == len(pdbids) == len(ca_icodes) == len(chain_letters)):
                raise ValueError(
                    f"Inconsistent batch sizes: {batch_len}, {len(pdbids)}, {len(ca_icodes)}, {len(chain_letters)}"
                )

            # Write PDB file for each structure in batch
            for k, pdbid in enumerate(pdbids):
                protein_dir = os.path.join(batched_data["out_dir"], pdbid)
                pdb_file = os.path.join(protein_dir, f"{pdbid}_pack_{i}.pdb")

                # Prepare B-factors (all zeros)
                b_factors = torch.zeros_like(result_batch["final_X"][k][:, :, 0])

                # Write PDB file
                write_full_PDB(
                    pdb_file,
                    result_batch["final_X"][k].detach().cpu().numpy(),
                    result_batch["atom14_mask"][k].detach().cpu().numpy(),
                    b_factors.detach().cpu().numpy(),
                    result_batch["R_idx"][k].detach().cpu().numpy(),
                    chain_letters[k],
                    result_batch["S"][k].detach().cpu().numpy(),
                    icodes=ca_icodes[k],
                )

                packed_files_list.append(pdb_file)

    return packed_files_list


def _group_files_by_pdbid(packed_files_list):
    """Group packed files by their protein filename."""
    grouped_files = {}
    for file_path in packed_files_list:
        # Extract the full protein filename (e.g., "1a28_seed0_pocket_backbone")
        protein_filename = os.path.basename(os.path.dirname(file_path))
        if protein_filename not in grouped_files:
            grouped_files[protein_filename] = []
        grouped_files[protein_filename].append(file_path)

    return list(grouped_files.values())
