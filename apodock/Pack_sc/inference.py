import os
from dataclasses import dataclass
from typing import Dict, Optional
from torch_geometric.utils import to_dense_batch
from apodock.utils import ModelManager, set_random_seed, logger
from apodock.Pack_sc.inference_utils import (
    cluster_pockets,
    get_letter_codes,
    pro2feature_dict,
    PackDataLoader,
    _move_batch_to_device,
    _process_ligand_features,
    _update_results_with_batch_data,
    _perform_resampling,
    _create_packing_dataset,
    _batch_input_data,
    _batch_residue_info,
    _clear_gpu_memory,
    _write_pdb_files,
    _group_files_by_pdbid,
)


@dataclass
class PackingConfig:
    """Configuration for protein side-chain packing.

    This class centralizes all parameters needed for the packing process,
    making it easier to manage and pass around parameters.
    """

    batch_size: int = 1
    number_of_packs_per_design: int = 5
    temperature: float = 0.1
    n_recycle: int = 3
    resample: bool = False
    resample_args: Optional[Dict] = None
    apo2holo: bool = False

    def __post_init__(self):
        """Initialize default values if needed."""
        if self.resample_args is None:
            self.resample_args = {}


def infer(
    model,
    dataloader,
    device,
    temperature=1.0,
    n_recycle=3,
    resample=True,
    resample_args=None,
):
    """
    Perform inference with the packing model.

    Args:
        model: The packing model
        dataloader: DataLoader with protein-ligand data
        device: Device to run inference on ('cpu', 'cuda', etc.)
        temperature: Temperature for sampling (default: 1.0)
        n_recycle: Number of recycling iterations (default: 3)
        resample: Whether to perform resampling (default: True)
        resample_args: Arguments for resampling (default: None)

    Returns:
        List of dictionaries with inference results

    Raises:
        RuntimeError: If inference fails
    """
    if resample_args is None:
        resample_args = {}

    model.eval()
    results_list = []

    try:
        for batch_idx, batch in enumerate(dataloader):
            # Process batch data
            try:
                # Move data to device
                batch = _move_batch_to_device(batch, device)

                # Get protein features
                feature_dict = pro2feature_dict(batch["protein"])

                # Process ligand features
                lig_feat = _process_ligand_features(model, batch["ligand"])

                # Convert to dense batch
                lig_batch, lig_mask = to_dense_batch(
                    lig_feat, batch["ligand"].batch, fill_value=0
                )

                # Run model sampling
                results = model.sample(
                    lig_batch,
                    lig_mask,
                    feature_dict,
                    temperature=temperature,
                    n_recycle=n_recycle,
                )

                # Update results with batch data
                results = _update_results_with_batch_data(results, batch, feature_dict)

                # Move results to CPU
                results = {k: v.detach().cpu() for k, v in results.items()}

                # Perform resampling if requested
                if resample:
                    results = _perform_resampling(results, resample_args)

                results_list.append(results)

            except Exception as e:
                raise RuntimeError(
                    f"Error processing batch {batch_idx}: {str(e)}"
                ) from e

        return results_list

    except Exception as e:
        raise RuntimeError(f"Inference failed: {str(e)}") from e


def sample_xyz(
    ligand_list,
    pocket_list,
    protein_list,
    model,
    device,
    config=None,
    ligandmpnn_path=None,
):
    """
    Sample side-chain conformations for a list of proteins.

    Args:
        ligand_list: List of ligand file paths
        pocket_list: List of pocket file paths
        protein_list: List of protein file paths
        model: The packing model
        device: Device to run inference on ('cpu', 'cuda', etc.)
        config: PackingConfig object with packing parameters
        ligandmpnn_path: Path to LigandMPNN model

    Returns:
        List of result lists containing sampled conformations

    Raises:
        RuntimeError: If sampling fails
    """
    # Use default configuration if none provided
    if config is None:
        config = PackingConfig()

    try:
        # Create dataset
        dataset = _create_packing_dataset(
            ligand_list, pocket_list, protein_list, ligandmpnn_path, config.apo2holo
        )

        # Create dataloader
        dataloader = PackDataLoader(
            dataset, batch_size=config.batch_size, shuffle=False
        )

        # Run inference for each packing iteration
        results_list = []
        for i in range(config.number_of_packs_per_design):
            results = infer(
                model,
                dataloader,
                device,
                temperature=config.temperature,
                n_recycle=config.n_recycle,
                resample=config.resample,
                resample_args=config.resample_args,
            )
            results_list.append(results)

        return results_list

    except Exception as e:
        raise RuntimeError(f"Failed to sample conformations: {str(e)}") from e


def write_pdbs(
    ligand_list,
    pocket_list,
    protein_list,
    model,
    device,
    config=None,
    ligandmpnn_path=None,
    out_dir="./packed_output",
):
    """
    Generate and write packed PDB structures.

    Args:
        ligand_list: List of ligand file paths
        pocket_list: List of pocket file paths
        protein_list: List of protein file paths
        model: The packing model
        device: Device to run inference on ('cpu', 'cuda', etc.)
        config: PackingConfig object with packing parameters
        ligandmpnn_path: Path to LigandMPNN model
        out_dir: Output directory for packed structures

    Returns:
        List of lists containing paths to generated PDB files, grouped by PDB ID

    Raises:
        RuntimeError: If PDB generation fails
    """
    # Use default configuration if none provided
    if config is None:
        config = PackingConfig()

    try:
        # Extract full protein filenames without extension
        protein_filenames = [
            os.path.splitext(os.path.basename(p))[0] for p in protein_list
        ]

        # Create output directories if they don't exist
        for protein_filename in protein_filenames:
            protein_dir = os.path.join(out_dir, protein_filename)
            os.makedirs(protein_dir, exist_ok=True)

        # Group inputs by batch size
        batched_data = _batch_input_data(
            ligand_list,
            pocket_list,
            protein_list,
            protein_filenames,  # Pass full filenames instead of PDB IDs
            out_dir,
            config.batch_size,
        )

        # Generate conformations
        results_list = sample_xyz(
            ligand_list,
            pocket_list,
            protein_list,
            model,
            device,
            config,
            ligandmpnn_path,
        )

        # Get residue information for PDB generation
        ca_icodes, chain_letters = get_letter_codes(pocket_list)
        batched_residue_info = _batch_residue_info(
            ca_icodes, chain_letters, config.batch_size
        )

        # Clear GPU memory
        _clear_gpu_memory(device)

        # Write PDB files
        packed_files = _write_pdb_files(
            results_list, batched_data, batched_residue_info
        )

        # Group files by protein filename
        grouped_files = _group_files_by_pdbid(packed_files)

        return grouped_files

    except Exception as e:
        # Clear GPU memory in case of error
        _clear_gpu_memory(device)
        raise RuntimeError(f"Failed to generate packed structures: {str(e)}") from e


def sc_pack(
    ligand_list,
    pocket_list,
    protein_list,
    model_sc,
    checkpoint_path,
    device,
    out_dir,
    config=None,
    ligandmpnn_path=None,
    num_clusters=6,
    random_seed=42,
    cleanup_intermediates=False,
):
    """
    Perform side-chain packing on protein pockets.

    Args:
        ligand_list: List of paths to ligand files
        pocket_list: List of paths to pocket files
        protein_list: List of paths to protein files
        model_sc: Side-chain packing model object
        checkpoint_path: Path to model checkpoint
        device: Device to run inference on ('cpu', 'cuda:0', etc.)
        out_dir: Output directory for packed structures
        config: PackingConfig object with packing parameters
        ligandmpnn_path: Path to LigandMPNN model
        num_clusters: Number of clusters for pocket clustering (default: 6)
        random_seed: Random seed for reproducibility (default: 42)
        cleanup_intermediates: Whether to clean up intermediate results (default: False)

    Returns:
        List of lists containing paths to clustered packed structures

    Raises:
        ValueError: If input validation fails
        RuntimeError: If packing or clustering fails
    """

    # Set random seed for reproducibility
    set_random_seed(random_seed, log=True)

    # Use default configuration if none provided
    if config is None:
        config = PackingConfig()

    # Input validation
    if not ligand_list or not pocket_list or not protein_list:
        raise ValueError("Empty input list provided")

    if len(ligand_list) != len(pocket_list) or len(ligand_list) != len(protein_list):
        raise ValueError(
            f"Mismatched input lists: ligands ({len(ligand_list)}), "
            f"pockets ({len(pocket_list)}), proteins ({len(protein_list)})"
        )

    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        logger.info(f"Starting side-chain packing for {len(ligand_list)} inputs...")

        # Step 1: Load model for inference
        model = ModelManager.load_model(model_sc, checkpoint_path, device)

        # Step 2: Generate packed structures
        logger.info(
            f"Generating {config.number_of_packs_per_design} packed structures per input..."
        )
        packed_files_list = write_pdbs(
            ligand_list=ligand_list,
            pocket_list=pocket_list,
            protein_list=protein_list,
            model=model,
            device=device,
            config=config,
            ligandmpnn_path=ligandmpnn_path,
            out_dir=out_dir,
        )

        # Step 3: Cluster the packed structures
        logger.info(
            f"Clustering packed structures into {num_clusters} clusters per input..."
        )
        cluster_packs_list = []

        for i, packed_files in enumerate(packed_files_list):
            if not packed_files:
                logger.warning(
                    f"Warning: No packed files generated for input {i+1}. Skipping."
                )
                cluster_packs_list.append([])
                continue

            # Filter out any non-packed structures (i.e., original backbone)
            packed_files = [f for f in packed_files if "_pack_" in os.path.basename(f)]
            if not packed_files:
                logger.warning(
                    f"Warning: No valid packed structures found for input {i+1}. Skipping."
                )
                cluster_packs_list.append([])
                continue

            cluster_packs = cluster_pockets(
                packed_files, num_clusters=num_clusters, random_seed=random_seed
            )
            cluster_packs_list.append(cluster_packs)

            # Clean up intermediate files if requested
            if cleanup_intermediates:
                # Create a set of directories that will need cleanup
                cleanup_dirs = set()

                # Remove packed files that weren't selected during clustering
                selected_files = set()
                for cluster_packs in cluster_packs_list:
                    selected_files.update(cluster_packs)

                # Find all packed files and remove those not in selected_files
                for file_path in packed_files:
                    if file_path not in selected_files:
                        try:
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                logger.info(
                                    f"Removed intermediate packed structure: {file_path}"
                                )
                                # Add the directory to our cleanup list
                                cleanup_dirs.add(os.path.dirname(file_path))
                        except Exception as e:
                            logger.warning(
                                f"Warning: Failed to remove {file_path}: {str(e)}"
                            )

                # Now check if any directories are empty and remove them
                for dir_path in cleanup_dirs:
                    try:
                        if os.path.exists(dir_path) and os.path.isdir(dir_path):
                            # Only remove directory if it contains no files or only contains log files
                            files = os.listdir(dir_path)
                            if not files or all(f.endswith(".log") for f in files):
                                import shutil

                                shutil.rmtree(dir_path)
                    except Exception as e:
                        logger.warning(
                            f"Warning: Failed to remove directory {dir_path}: {str(e)}"
                        )

        # Clean up GPU memory
        ModelManager.cleanup_gpu_memory(device)

        logger.info("Side-chain packing completed successfully")
        return cluster_packs_list

    except Exception as e:
        # Clean up GPU memory in case of error
        ModelManager.cleanup_gpu_memory(device)
        raise RuntimeError(f"Failed to perform side-chain packing: {str(e)}") from e
