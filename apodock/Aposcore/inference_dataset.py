import os
import sys
import numpy as np
import torch
import traceback
from itertools import repeat
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch
from torch_scatter import scatter_add
from Bio.PDB import PDBParser
from apodock.Aposcore.dataset_Aposcore import mol2graph, get_pro_coord
from apodock.Aposcore.utils import (
    get_clean_res_list,
    get_protein_feature,
)
from apodock.utils import ModelManager, set_random_seed, logger
from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoringConfig:
    """Configuration for protein-ligand scoring.

    This class centralizes all parameters needed for the scoring process,
    making it easier to manage and pass around parameters.
    """

    batch_size: int = 64
    dis_threshold: float = 5.0
    output_dir: Optional[str] = None

    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.output_dir is not None and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)


def read_sdf_file(mol_file, save_mols=False):
    """
    This function reads a SDF file containing multiple molecules and returns a dict of RDKit molecule objects.
    """
    if not os.path.exists(mol_file):
        sys.exit(f"The MOL2 file {mol_file} does not exist")
    mol_file_name = os.path.basename(mol_file).split(".")[0]

    # Use single-threaded SDMolSupplier instead of MultithreadedSDMolSupplier
    # to avoid potential thread synchronization issues
    supplier = Chem.SDMolSupplier(mol_file, removeHs=True, sanitize=True)
    molecules = []
    molecules_name = []
    for i, mol in enumerate(supplier):
        if mol is not None:
            molecules_name.append(mol_file_name + "_" + str(i))
            molecules.append(mol)

    if len(molecules) == 0:
        logger.error(f"No molecules pass the rdkit sanitization in {mol_file}")
        if save_mols:
            return None, None

        logger.info(f"Trying again without sanitization for {mol_file}...")
        supplier = Chem.SDMolSupplier(mol_file, removeHs=True, sanitize=False)
        for i, mol in enumerate(supplier):
            if mol is not None:
                molecules_name.append(mol_file_name + "_" + str(i))
                molecules.append(mol)
    return molecules, molecules_name


def get_graph_data_l(mol):
    """Convert ligand molecule to graph data"""
    atom_num_l = mol.GetNumAtoms()
    x_l, edge_index_l, edge_features_l = mol2graph(mol)
    pos_l = torch.FloatTensor(mol.GetConformers()[0].GetPositions())
    c_size_l = atom_num_l
    data_l = Data(
        x=x_l,
        edge_index=edge_index_l,
        edge_attr=edge_features_l,
        pos=pos_l,
    )
    data_l.__setitem__("c_size", torch.LongTensor([c_size_l]))
    return data_l


def get_graph_data_p(pocket_path):
    """Convert protein pocket to graph data"""
    pocket_id = os.path.basename(pocket_path)
    parser = PDBParser(QUIET=True)
    res_list_feature = get_clean_res_list(
        parser.get_structure(pocket_id, pocket_path).get_residues(),
        verbose=False,
        ensure_ca_exist=True,
    )
    x_aa, seq, node_s, node_v, edge_index, edge_s, edge_v = get_protein_feature(
        res_list_feature
    )

    c_size_aa = len(seq)
    res_list_coord = [
        res
        for res in parser.get_structure(pocket_id, pocket_path).get_residues()
        if res in res_list_feature
    ]

    if len(res_list_coord) != len(res_list_feature):
        raise ValueError(
            f"The length of res_list_coord({len(res_list_coord)}) is not equal to the length of res_list_feature({len(res_list_feature)})"
        )

    pos_p, _, _ = get_pro_coord(res_list_coord)

    data_aa = Data(
        x_aa=x_aa,
        seq=seq,
        node_s=node_s,
        node_v=node_v,
        edge_index=edge_index,
        edge_s=edge_s,
        edge_v=edge_v,
        pos=pos_p,
    )
    data_aa.__setitem__("c_size", torch.LongTensor([c_size_aa]))
    return data_aa


def mol2graphs_dock(mol_list, pocket):
    """Convert molecules and pocket to graph data for docking"""
    graph_data_list_l = []
    graph_data_name_l = []

    # Process ligands with detailed progress reporting
    for idx, mol in enumerate(mol_list):
        try:
            # Check if molecule is valid
            if mol is None:
                logger.info(f"    Skipping molecule {idx+1}: Molecule is None")
                continue

            if mol.GetNumAtoms() == 0:
                logger.info(f"    Skipping molecule {idx+1}: Molecule has no atoms")
                continue

            # Check if molecule has 3D coordinates
            try:
                conf = mol.GetConformer()
                if conf.Is3D() == False:
                    logger.warning(
                        f"    Warning: Molecule {idx+1} does not have 3D coordinates"
                    )
            except:
                logger.info(f"    Skipping molecule {idx+1}: No conformer available")
                continue

            # Convert to graph
            try:
                graph_mol = get_graph_data_l(mol)
                graph_data_list_l.append(graph_mol)
                graph_data_name_l.append(mol)
            except Exception as e:
                logger.error(
                    f"    Error converting molecule {idx+1} to graph: {str(e)}"
                )
                logger.error(traceback.format_exc())
                continue

        except Exception as e:
            logger.error(f"    Unexpected error processing molecule {idx+1}: {str(e)}")
            continue

    # Check if we have any valid ligands
    if len(graph_data_list_l) == 0:
        raise ValueError("No valid ligand molecules could be converted to graphs")

    # Process pocket with detailed error handling
    try:
        # Check if pocket file exists
        if not os.path.exists(pocket):
            raise FileNotFoundError(f"Pocket file not found: {pocket}")

        # Convert pocket to graph
        graph_data_aa = get_graph_data_p(pocket)
        graph_data_list_aa = list(repeat(graph_data_aa, len(graph_data_list_l)))

        return graph_data_list_l, graph_data_name_l, graph_data_list_aa

    except Exception as e:
        logger.error(f"Error processing pocket {pocket}: {str(e)}")
        logger.error(traceback.format_exc())
        raise


class PLIDataLoader(DataLoader):
    """Custom DataLoader for protein-ligand interaction data"""

    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


class Dataset_infer(Dataset):
    """Dataset class for inference"""

    def __init__(self, sdf_list_path, pocket_list_path):
        self.sdf_list_path = sdf_list_path
        self.pocket_list_path = pocket_list_path
        self.graph_l_list = []
        self.graph_aa_list = []
        self.graph_name_list = []
        self._pre_process()

    def _pre_process(self):
        for i, (mol_file, pocket) in enumerate(
            zip(self.sdf_list_path, self.pocket_list_path)
        ):
            try:
                # Check if files exist
                if not os.path.exists(mol_file):
                    logger.warning(f"Warning: SDF file does not exist: {mol_file}")
                    continue
                if not os.path.exists(pocket):
                    logger.warning(f"Warning: Pocket file does not exist: {pocket}")
                    continue

                # Read molecules with improved error handling
                try:
                    mol, _ = read_sdf_file(mol_file, save_mols=False)
                    if mol is None or len(mol) == 0:
                        logger.warning(
                            f"Warning: Could not read any valid molecules from {mol_file}"
                        )
                        continue
                except Exception as e:
                    logger.error(f"Error reading molecules from {mol_file}: {str(e)}")
                    continue

                # Convert molecules and pocket to graph data
                try:
                    graph_data_list_l, graph_data_name_l, graph_data_list_aa = (
                        mol2graphs_dock(mol, pocket)
                    )
                except Exception as e:
                    logger.error(
                        f"Error converting molecules to graph data for {mol_file}: {str(e)}"
                    )
                    logger.error(traceback.format_exc())
                    continue

                # Extend lists one at a time to avoid memory spikes
                self.graph_l_list.extend(graph_data_list_l)
                self.graph_aa_list.extend(graph_data_list_aa)
                self.graph_name_list.extend(graph_data_name_l)

                # Clear unnecessary references to free memory
                del graph_data_list_l
                del graph_data_list_aa
                del graph_data_name_l
                del mol

                # Force garbage collection to free memory
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Unexpected error processing {mol_file}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        # Check if we have any valid data
        if len(self.graph_l_list) == 0:
            logger.warning(
                "Warning: No valid structures were processed. Dataset is empty."
            )
            raise RuntimeError("Failed to process any valid structures for inference.")

    def __del__(self):
        """Explicit cleanup when the dataset is destroyed."""
        self.graph_l_list.clear()
        self.graph_aa_list.clear()
        self.graph_name_list.clear()

    def __getitem__(self, idx):
        return self.graph_l_list[idx], self.graph_aa_list[idx]

    def collate_fn(self, data_list):
        batchA = Batch.from_data_list([data[0] for data in data_list])
        batchB = Batch.from_data_list([data[1] for data in data_list])
        batch = {"ligand_features": batchA, "protein_features": batchB}
        return batch

    def __len__(self):
        return len(self.graph_l_list)


def val(model, dataloader, device, dis_threshold=5.0):
    """
    Validation function for calculating probabilities from the MDN model.

    Args:
        model: The Aposcore model in evaluation mode
        dataloader: DataLoader containing the data to evaluate
        device: Device to run inference on ('cpu', 'cuda', etc.)
        dis_threshold: Distance threshold for filtering interactions (default: 5.0)

    Returns:
        numpy.ndarray: Array of probabilities for each input

    Raises:
        RuntimeError: If no valid predictions could be generated
    """
    model.eval()  # Ensure model is in evaluation mode
    probs = []
    total_batches = len(dataloader)

    for batch_idx, data in enumerate(dataloader):
        try:
            logger.info(f"Processing batch {batch_idx+1}/{total_batches}...")

            # Check batch data
            if "ligand_features" not in data or "protein_features" not in data:
                logger.warning(
                    f"Warning: Batch {batch_idx+1} is missing required features"
                )
                continue
            with torch.no_grad():
                data["ligand_features"] = data["ligand_features"].to(device)
                data["protein_features"] = data["protein_features"].to(device)
                pi, sigma, mu, dist, batch = model(data)
                prob = model.calculate_probablity(pi, sigma, mu, dist)
                prob[torch.where(dist > dis_threshold)[0]] = 0.0

                batch = batch.to(device)
                probx = scatter_add(prob, batch, dim=0, dim_size=batch.unique().size(0))
                probs.append(probx.cpu().numpy())

        except Exception as e:
            logger.error(f"Error processing batch {batch_idx+1}: {str(e)}")
            logger.error(traceback.format_exc())
            continue

    if not probs:
        raise RuntimeError("No valid predictions generated")

    # Concatenate all batch results
    print("Finalizing predictions...")
    pred = np.concatenate(probs)
    print(f"Evaluation complete. Generated {len(pred)} predictions.")
    return pred


def get_mdn_score(
    sdf_files,
    pocket_files,
    model,
    checkpoint_path,
    device,
    config=None,
    random_seed=42,
):
    """
    Calculate Docking scores for docked poses.

    Args:
        sdf_files: List of SDF file paths containing docked poses
        pocket_files: List of protein pocket file paths
        model: The Aposcore model object
        checkpoint_path: Path to the model checkpoint file
        device: Device to run inference on ('cpu', 'cuda', etc.)
        config: ScoringConfig object with scoring parameters (default: None)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        Dict[str, List[float]]: Dictionary mapping structure IDs to their pose scores
    """
    try:
        # Set random seed for reproducibility
        set_random_seed(random_seed, log=True)
        # Validate inputs
        if not sdf_files or not pocket_files:
            raise ValueError("Empty input files list provided")

        if len(sdf_files) != len(pocket_files):
            raise ValueError(
                f"Number of SDF files ({len(sdf_files)}) must match number of pocket files ({len(pocket_files)})"
            )

        # Load model for inference
        model = ModelManager.load_model(model, checkpoint_path, device)

        # Create dataset and dataloader with explicit cleanup
        dataset = None
        dataloader = None
        try:
            dataset = Dataset_infer(sdf_files, pocket_files)

            # Check if dataset is empty
            if len(dataset) == 0:
                logger.warning("Warning: Dataset is empty. No structures to score.")
                return {}

            dataloader = PLIDataLoader(
                dataset, batch_size=config.batch_size, shuffle=False
            )

            # Calculate scores
            all_scores = val(model, dataloader, device, config.dis_threshold)
            logger.info(f"Scoring completed. Got {len(all_scores)} scores.")

            # Map scores back to their structures
            scores_dict = {}
            score_idx = 0

            for sdf_file, pocket_file in zip(sdf_files, pocket_files):
                if "_pack_" not in os.path.basename(pocket_file):
                    logger.info(
                        f"Skipping non-packed structure: {os.path.basename(pocket_file)}"
                    )
                    continue

                try:
                    mols, _ = read_sdf_file(sdf_file, save_mols=False)
                    if mols is None:
                        logger.warning(
                            f"Warning: Could not read molecules from {sdf_file}"
                        )
                        continue

                    num_poses = len(mols)

                    if score_idx + num_poses <= len(all_scores):
                        structure_id = os.path.splitext(os.path.basename(pocket_file))[
                            0
                        ]
                        scores_dict[structure_id] = all_scores[
                            score_idx : score_idx + num_poses
                        ].tolist()
                        score_idx += num_poses
                    else:
                        logger.warning(
                            f"Warning: Not enough scores for {pocket_file}. Need {num_poses} scores but only {len(all_scores) - score_idx} available."
                        )
                        break
                except Exception as e:
                    logger.error(f"Error processing {sdf_file}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

            return scores_dict

        except Exception as e:
            logger.error(f"Error in dataset/dataloader creation or scoring: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

        finally:
            # Explicit cleanup
            if dataloader is not None:
                del dataloader
            if dataset is not None:
                del dataset
            # Clean up GPU memory
            if torch.cuda.is_available():
                ModelManager.cleanup_gpu_memory(device)
                torch.cuda.empty_cache()

    except Exception as e:
        logger.error(f"Error in get_mdn_score: {str(e)}")
        logger.error(traceback.format_exc())
        # Catch and re-raise with more context
        raise RuntimeError(f"Failed to calculate MDN scores: {str(e)}") from e
