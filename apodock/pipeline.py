import os
from typing import List
import numpy as np
import torch

from apodock.config import PipelineConfig
from apodock.utils import ApoDockError, logger, set_random_seed
from apodock.pocket_extractor import PocketExtractor
from apodock.docking_engine import DockingEngine
from apodock.results_processor import ResultsProcessor

from apodock.Pack_sc.inference import sc_pack
from apodock.Pack_sc.Packnn import Pack
from apodock.Aposcore.inference_dataset import (
    get_mdn_score,
    read_sdf_file,
    ScoringConfig,
)
from apodock.Aposcore.Aposcore import Aposcore


class DockingPipeline:
    """Main docking pipeline class that orchestrates the docking process."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize the docking pipeline.

        Args:
            config: Configuration for the docking pipeline
        """
        self.config = config

        # Set random seed for reproducibility
        random_seed = config.random_seed
        set_random_seed(random_seed)

        logger.info(f"Setting random seed to {random_seed} for reproducibility")

        self.pocket_extractor = PocketExtractor(config.pocket_distance)
        self.docking_engine = DockingEngine(config.docking_config, random_seed)
        self.results_processor = ResultsProcessor(config.top_k)

        # Create uninitialized models
        self.pack_model = None
        self.score_model = None

        logger.info("Initializing machine learning models...")

        # Initialize Pack model if needed
        if config.use_packing:
            logger.info(f"Creating Pack model for side-chain packing...")
            self.pack_model = Pack(recycle_strategy="sample")
            # Note: We don't load weights here - they'll be loaded during sc_pack

        # Always initialize Aposcore model
        logger.info(f"Creating Aposcore model for pose scoring...")
        self.score_model = Aposcore(
            35,
            hidden_dim=config.aposcore_config.hidden_dim,
            num_heads=config.aposcore_config.num_heads,
            dropout=config.aposcore_config.dropout,
            crossAttention=config.aposcore_config.cross_attention,
            atten_active_fuc=config.aposcore_config.attention_activation,
            num_layers=config.aposcore_config.num_layers,
            interact_type=config.aposcore_config.interaction_type,
            n_gaussians=config.aposcore_config.n_gaussians,
        )
        # Note: We don't load weights here - they'll be loaded during get_mdn_score

    def run_pocket_screening(
        self,
        ligand_list: List[str],
        protein_list: List[str],
        ref_lig_list: List[str],
        save_poses: bool = False,
        output_scores_file: str = None,
        rank_by: str = "aposcore",
    ) -> dict:
        """
        Run the docking pipeline optimized for pocket screening.

        This method focuses on obtaining scores for pocket designs rather than
        generating detailed pose files. It returns the best score for each pocket
        and optionally saves these scores to a file.

        Args:
            ligand_list: List of paths to ligand files
            protein_list: List of paths to protein files
            ref_lig_list: List of paths to reference ligand files
            save_poses: Whether to save the docked poses (default: False)
            output_scores_file: Path to save the scores (default: None)
            rank_by: Which score to use for ranking (default: "aposcore")
                Options: "aposcore", "gnina_affinity", "gnina_cnn_score", "gnina_cnn_affinity"

        Returns:
            Dictionary mapping protein IDs to their best scores
        """
        # Use the configuration from self.config
        config = self.config

        # Extract protein IDs for directory organization
        protein_ids = [os.path.basename(p).split("_protein")[0] for p in protein_list]

        # Step 1: Extract pockets from proteins using reference ligands
        logger.info("Extracting pockets from proteins...")
        pocket_list = self.pocket_extractor.extract_pockets(
            protein_list, ref_lig_list, config.output_dir
        )

        # Dictionary to store best scores for each protein
        best_scores = {}
        # Dictionary to track best structure files for each protein
        best_structures = {}

        try:
            # Decide on strategy: packing or direct docking
            if config.use_packing:
                # Step 2a: With packing - do side chain packing first
                logger.info("Running side chain packing...")
                # Create a PackingConfig object from our existing config parameters
                from apodock.Pack_sc.inference import PackingConfig

                packing_config = PackingConfig(
                    batch_size=config.pack_config.packing_batch_size,
                    number_of_packs_per_design=config.pack_config.packs_per_design,
                    temperature=config.pack_config.temperature,
                    n_recycle=3,  # Default value
                    resample=True,  # Default value
                    apo2holo=False,  # Explicitly set to False
                )

                cluster_packs_list = sc_pack(
                    ligand_list,
                    pocket_list,
                    protein_list,
                    self.pack_model,
                    config.pack_config.checkpoint_path,
                    config.pack_config.device,
                    config.output_dir,
                    config=packing_config,
                    ligandmpnn_path=config.pack_config.ligandmpnn_path,
                    num_clusters=config.pack_config.num_clusters,
                    random_seed=config.random_seed,
                    cleanup_intermediates=not save_poses,  # In screening mode, use the save_poses parameter
                )

                # Step 2b: With packing - dock ligands to packed proteins
                logger.info("Docking ligands to packed proteins...")
                docking_results = self.docking_engine.dock_to_packed_proteins(
                    ligand_list, cluster_packs_list, ref_lig_list, config.output_dir
                )
                docked_ligands, packed_proteins_used = (
                    docking_results  # Unpack the results
                )
            else:
                # Step 2c: Without packing - dock ligands directly to extracted pockets
                logger.info("Docking ligands to protein pockets...")
                docked_ligands = self.docking_engine.dock_to_pockets(
                    ligand_list, pocket_list, ref_lig_list, config.output_dir
                )

            # Step 3: Score docked poses
            logger.info("Scoring docked poses...")
            scoring_config = ScoringConfig(
                batch_size=64,  # Default batch size
                dis_threshold=5.0,  # Default distance threshold
                output_dir=config.output_dir,
            )

            # Prepare protein structures for scoring
            if config.use_packing:
                # When using packing, use the exact packed protein structures that were used for docking
                logger.info(
                    f"Using {len(packed_proteins_used)} packed protein structures for scoring"
                )
                # Verify we have matching lengths
                if len(packed_proteins_used) != len(docked_ligands):
                    logger.error(
                        f"Mismatch between docked ligands ({len(docked_ligands)}) and packed proteins ({len(packed_proteins_used)})"
                    )
                    raise ApoDockError(
                        "Inconsistent number of docked poses and protein structures"
                    )
                scoring_pocket_files = packed_proteins_used
            else:
                # For direct docking, there should be a 1:1 relationship between docked ligands and pockets
                if len(docked_ligands) != len(pocket_list):
                    logger.warning(
                        f"Mismatch in direct docking: {len(docked_ligands)} docked ligands but {len(pocket_list)} pockets"
                    )
                    # If there's a mismatch, use the shortest list length to avoid index errors
                    min_len = min(len(docked_ligands), len(pocket_list))
                    docked_ligands = docked_ligands[:min_len]
                    scoring_pocket_files = pocket_list[:min_len]
                else:
                    scoring_pocket_files = pocket_list

            # Get scores for all poses
            scores = get_mdn_score(
                docked_ligands,
                scoring_pocket_files,
                self.score_model,
                config.aposcore_config.checkpoint_path,
                config.aposcore_config.device,
                config=scoring_config,
                random_seed=config.random_seed,
            )

            # Process scores to get the best score for each protein and track the best structures
            if isinstance(scores, np.ndarray):
                logger.info(f"Received {len(scores)} scores from get_mdn_score")

                # Map scores to their corresponding proteins
                score_index = 0
                for i, sdf_file in enumerate(docked_ligands):
                    # Extract protein ID from the SDF file path or directory
                    protein_dir = os.path.dirname(sdf_file)
                    protein_id = os.path.basename(protein_dir)

                    # Use the protein ID from the list if it's available
                    if i < len(protein_ids):
                        protein_id = protein_ids[i]

                    # Count molecules in this SDF file
                    mols, _ = read_sdf_file(sdf_file, save_mols=True)
                    if mols is None:
                        continue

                    num_mols = len(mols)
                    if score_index + num_mols <= len(scores):
                        # Get scores for molecules in this file
                        file_scores = scores[score_index : score_index + num_mols]
                        score_index += num_mols
                    else:
                        # If we don't have enough scores, use available ones and log warning
                        remaining = len(scores) - score_index
                        if remaining > 0:
                            file_scores = scores[score_index:]
                            score_index = len(scores)
                        else:
                            # No scores left, set to empty list
                            file_scores = []
                        logger.warning(
                            f"Not enough scores for file {sdf_file}: needed {num_mols}, got {len(file_scores)}"
                        )

                    # Update best score for this protein
                    if len(file_scores) > 0:
                        current_best = best_scores.get(protein_id, {})
                        if not current_best:
                            current_best = {
                                "aposcore": -float("inf"),
                                "gnina_affinity": float("inf"),
                                "gnina_cnn_score": -float("inf"),
                                "gnina_cnn_affinity": float("inf"),
                            }

                        # Get the best ApoScore
                        best_aposcore_idx = np.argmax(file_scores)
                        best_aposcore = float(file_scores[best_aposcore_idx])

                        # Try to get GNINA scores
                        from apodock.utils import extract_gnina_scores_from_file

                        gnina_scores = extract_gnina_scores_from_file(sdf_file)

                        if gnina_scores:
                            # Find the best scores for each GNINA metric
                            best_affinity = min(
                                [s["affinity"] for s in gnina_scores],
                                default=float("inf"),
                            )
                            best_cnn_score = max(
                                [s["cnn_score"] for s in gnina_scores],
                                default=-float("inf"),
                            )
                            best_cnn_affinity = min(
                                [s["cnn_affinity"] for s in gnina_scores],
                                default=float("inf"),
                            )

                            # Update if better than current best
                            update_best = False
                            if best_aposcore > current_best["aposcore"]:
                                current_best["aposcore"] = best_aposcore
                                update_best = rank_by == "aposcore"
                            if best_affinity < current_best["gnina_affinity"]:
                                current_best["gnina_affinity"] = best_affinity
                                update_best = update_best or (
                                    rank_by == "gnina_affinity"
                                )
                            if best_cnn_score > current_best["gnina_cnn_score"]:
                                current_best["gnina_cnn_score"] = best_cnn_score
                                update_best = update_best or (
                                    rank_by == "gnina_cnn_score"
                                )
                            if best_cnn_affinity < current_best["gnina_cnn_affinity"]:
                                current_best["gnina_cnn_affinity"] = best_cnn_affinity
                                update_best = update_best or (
                                    rank_by == "gnina_cnn_affinity"
                                )

                            # Store best structures if we've updated the best score
                            if update_best and i < len(scoring_pocket_files):
                                best_structures[protein_id] = {
                                    "protein": scoring_pocket_files[i],
                                    "ligand": sdf_file,
                                    "score": current_best[rank_by],
                                }
                        else:
                            # If no GNINA scores, just update ApoScore
                            if best_aposcore > current_best["aposcore"]:
                                current_best["aposcore"] = best_aposcore
                                # Store best structures
                                if i < len(scoring_pocket_files):
                                    best_structures[protein_id] = {
                                        "protein": scoring_pocket_files[i],
                                        "ligand": sdf_file,
                                        "score": best_aposcore,
                                    }

                        best_scores[protein_id] = current_best
                        logger.info(f"Best scores for {protein_id}: {current_best}")

            # Save scores to file if requested
            if output_scores_file:
                scores_path = os.path.join(config.output_dir, output_scores_file)

                # Determine sort order based on rank_by parameter
                if rank_by == "aposcore":
                    reverse = True  # Higher is better
                    sort_key = lambda x: x[1].get("aposcore", -float("inf"))
                elif rank_by == "gnina_affinity":
                    reverse = False  # Lower is better
                    sort_key = lambda x: x[1].get("gnina_affinity", float("inf"))
                elif rank_by == "gnina_cnn_score":
                    reverse = True  # Higher is better
                    sort_key = lambda x: x[1].get("gnina_cnn_score", -float("inf"))
                elif rank_by == "gnina_cnn_affinity":
                    reverse = False  # Lower is better
                    sort_key = lambda x: x[1].get("gnina_cnn_affinity", float("inf"))
                else:
                    # Default to ApoScore if invalid rank_by
                    logger.warning(
                        f"Invalid rank_by parameter: {rank_by}. Using aposcore instead."
                    )
                    reverse = True
                    sort_key = lambda x: x[1].get("aposcore", -float("inf"))

                # Sort the results
                sorted_items = sorted(
                    best_scores.items(), key=sort_key, reverse=reverse
                )

                with open(scores_path, "w") as f:
                    # Write header with all score types
                    f.write(
                        "Rank,Protein_ID,ApoScore,GNINA_Affinity,GNINA_CNN_Score,GNINA_CNN_Affinity\n"
                    )

                    # Write sorted results with rank
                    for rank, (protein_id, score_dict) in enumerate(sorted_items, 1):
                        # Get values with defaults for missing scores
                        aposcore = score_dict.get("aposcore", "N/A")
                        gnina_affinity = score_dict.get("gnina_affinity", "N/A")
                        gnina_cnn_score = score_dict.get("gnina_cnn_score", "N/A")
                        gnina_cnn_affinity = score_dict.get("gnina_cnn_affinity", "N/A")

                        # Write the scores
                        f.write(
                            f"{rank},{protein_id},{aposcore},{gnina_affinity},{gnina_cnn_score},{gnina_cnn_affinity}\n"
                        )

                logger.info(f"Saved best scores to {scores_path} (ranked by {rank_by})")

            # Clean up intermediate files and save only the best structures if screening mode
            if not save_poses:
                # Get the overall best protein ID based on ranking
                if len(sorted_items) > 0:
                    best_protein_id = sorted_items[0][0]
                    logger.info(f"Best overall protein ID based on {rank_by}: {best_protein_id}")
                
                # Track directories that might become empty after cleanup
                cleaned_dirs = set()
                
                # First, save the best structures for each basic protein ID
                # (we need to extract base protein IDs without pack numbers)
                base_protein_ids = {}
                for protein_id in best_structures.keys():
                    # Extract base ID (removing _pack_XX suffix if present)
                    base_id = protein_id.split("_pack_")[0] if "_pack_" in protein_id else protein_id
                    
                    # For each base ID, keep track of its best variant
                    if base_id not in base_protein_ids or sort_key((base_id, best_scores[protein_id])) > sort_key((base_id, best_scores[base_protein_ids[base_id]])):
                        base_protein_ids[base_id] = protein_id
                
                logger.info(f"Best variant for each base protein: {base_protein_ids}")
                
                # Now save only the best variant for each base protein ID
                for base_id, best_variant_id in base_protein_ids.items():
                    best_structure = best_structures[best_variant_id]
                    protein_dir = os.path.join(config.output_dir, base_id)
                    
                    # Ensure directory exists
                    os.makedirs(protein_dir, exist_ok=True)
                    
                    # Keep only the best protein structure
                    best_protein = best_structure["protein"]
                    best_ligand = best_structure["ligand"]
                    
                    # Create best structure filenames
                    best_protein_filename = f"{base_id}_best_{rank_by}_protein.pdb"
                    best_ligand_filename = f"{base_id}_best_{rank_by}_ligand.sdf"
                    
                    # Copy the best structures with new names
                    import shutil
                    try:
                        shutil.copy2(best_protein, os.path.join(protein_dir, best_protein_filename))
                        shutil.copy2(best_ligand, os.path.join(protein_dir, best_ligand_filename))
                        logger.info(f"Saved best structures for {base_id}: {best_protein_filename}, {best_ligand_filename}")
                    except Exception as e:
                        logger.warning(f"Failed to save best structures for {base_id}: {str(e)}")
                
                # Clean up ALL packed protein files across the entire output directory
                if config.use_packing:
                    # Find all packed protein files
                    all_pack_files = []
                    for root, _, files in os.walk(config.output_dir):
                        for f in files:
                            # Match any file with _pack_ in the name, except for the saved best files
                            if "_pack_" in f and not f.endswith(f"_best_{rank_by}_protein.pdb") and not f.endswith(f"_best_{rank_by}_ligand.sdf"):
                                file_path = os.path.join(root, f)
                                all_pack_files.append(file_path)
                                # Track directory for potential cleanup
                                cleaned_dirs.add(os.path.dirname(file_path))
                    
                    # Remove all packed files except the best ones we copied
                    for packed_file in all_pack_files:
                        try:
                            os.remove(packed_file)
                            logger.debug(f"Removed intermediate structure: {packed_file}")
                        except Exception as e:
                            logger.warning(f"Failed to remove file {packed_file}: {str(e)}")
                
                # Clean up pocket files
                self.pocket_extractor.cleanup_temp_files()
                
                # Clean up empty directories
                for dir_path in cleaned_dirs:
                    try:
                        # Check if directory exists and is empty
                        if os.path.exists(dir_path) and os.path.isdir(dir_path) and not os.listdir(dir_path):
                            os.rmdir(dir_path)
                            logger.debug(f"Removed empty directory: {dir_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove directory {dir_path}: {str(e)}")
            elif isinstance(scores, np.ndarray):
                # If save_poses is True, save all docked poses like in the original code
                # Convert scores to a format that the results processor can handle
                processed_scores = []
                score_index = 0
                for sdf_file in docked_ligands:
                    # Count molecules in this SDF file
                    mols, _ = read_sdf_file(sdf_file, save_mols=True)
                    if mols is None:
                        continue

                    num_mols = len(mols)
                    if score_index + num_mols <= len(scores):
                        # Get scores for molecules in this file
                        file_scores = scores[score_index : score_index + num_mols]
                        score_index += num_mols
                    else:
                        # If we don't have enough scores, use available ones and log warning
                        remaining = len(scores) - score_index
                        if remaining > 0:
                            file_scores = scores[score_index:]
                            score_index = len(scores)
                        else:
                            # No scores left, set to empty list
                            file_scores = []

                    processed_scores.append((sdf_file, file_scores))

                # Process results to save poses
                logger.info("Processing results to save poses...")
                self.results_processor.process(processed_scores, config.output_dir)

            logger.info(f"Pocket screening completed successfully.")
            return best_scores

        except Exception as e:
            logger.error(f"Error during pocket screening: {str(e)}")
            # Clean up GPU memory if available
            try:
                if torch.cuda.is_available() and config.pack_config.device.startswith(
                    "cuda"
                ):
                    from apodock.utils import ModelManager

                    ModelManager.cleanup_gpu_memory(config.pack_config.device)
                if (
                    torch.cuda.is_available()
                    and config.aposcore_config.device.startswith("cuda")
                ):
                    from apodock.utils import ModelManager

                    ModelManager.cleanup_gpu_memory(config.aposcore_config.device)
            except Exception as cleanup_error:
                logger.error(f"Error during GPU cleanup: {str(cleanup_error)}")

            raise

    def run(
        self, ligand_list: List[str], protein_list: List[str], ref_lig_list: List[str]
    ) -> List[str]:
        """
        Run the docking pipeline.

        Args:
            ligand_list: List of paths to ligand files
            protein_list: List of paths to protein files
            ref_lig_list: List of paths to reference ligand files

        Returns:
            List of paths to docked ligand files
        """
        # Use the configuration from self.config
        config = self.config

        # Step 1: Extract pockets from proteins using reference ligands
        logger.info("Extracting pockets from proteins...")
        pocket_list = self.pocket_extractor.extract_pockets(
            protein_list, ref_lig_list, config.output_dir
        )

        try:
            # Decide on strategy: packing or direct docking
            if config.use_packing:
                # Step 2a: With packing - do side chain packing first
                logger.info("Running side chain packing...")
                # Create a PackingConfig object from our existing config parameters
                from apodock.Pack_sc.inference import PackingConfig

                packing_config = PackingConfig(
                    batch_size=config.pack_config.packing_batch_size,
                    number_of_packs_per_design=config.pack_config.packs_per_design,
                    temperature=config.pack_config.temperature,
                    n_recycle=3,  # Default value
                    resample=True,  # Default value
                    apo2holo=False,  # Explicitly set to False
                )

                cluster_packs_list = sc_pack(
                    ligand_list,
                    pocket_list,
                    protein_list,
                    self.pack_model,
                    config.pack_config.checkpoint_path,
                    config.pack_config.device,
                    config.output_dir,
                    config=packing_config,
                    ligandmpnn_path=config.pack_config.ligandmpnn_path,
                    num_clusters=config.pack_config.num_clusters,
                    random_seed=config.random_seed,
                    cleanup_intermediates=config.screening_mode and not config.save_poses,  # Clean up intermediates in screening mode only
                )

                # Step 2b: With packing - dock ligands to packed proteins
                logger.info("Docking ligands to packed proteins...")
                docking_results = self.docking_engine.dock_to_packed_proteins(
                    ligand_list, cluster_packs_list, ref_lig_list, config.output_dir
                )
                docked_ligands, packed_proteins_used = (
                    docking_results  # Unpack the results
                )
            else:
                # Step 2c: Without packing - dock ligands directly to extracted pockets
                logger.info("Docking ligands to protein pockets...")
                docked_ligands = self.docking_engine.dock_to_pockets(
                    ligand_list, pocket_list, ref_lig_list, config.output_dir
                )

            # Step 3: Score docked poses
            logger.info("Scoring docked poses...")
            scoring_config = ScoringConfig(
                batch_size=64,  # Default batch size
                dis_threshold=5.0,  # Default distance threshold
                output_dir=config.output_dir,
            )

            # Prepare protein structures for scoring
            if config.use_packing:
                # When using packing, use the exact packed protein structures that were used for docking
                logger.info(
                    f"Using {len(packed_proteins_used)} packed protein structures for scoring"
                )
                # Verify we have matching lengths
                if len(packed_proteins_used) != len(docked_ligands):
                    logger.error(
                        f"Mismatch between docked ligands ({len(docked_ligands)}) and packed proteins ({len(packed_proteins_used)})"
                    )
                    raise ApoDockError(
                        "Inconsistent number of docked poses and protein structures"
                    )
                scoring_pocket_files = packed_proteins_used
            else:
                # For direct docking, there should be a 1:1 relationship between docked ligands and pockets
                if len(docked_ligands) != len(pocket_list):
                    logger.warning(
                        f"Mismatch in direct docking: {len(docked_ligands)} docked ligands but {len(pocket_list)} pockets"
                    )
                    # If there's a mismatch, use the shortest list length to avoid index errors
                    min_len = min(len(docked_ligands), len(pocket_list))
                    docked_ligands = docked_ligands[:min_len]
                    scoring_pocket_files = pocket_list[:min_len]
                else:
                    scoring_pocket_files = pocket_list

            scored_poses = get_mdn_score(
                docked_ligands,
                scoring_pocket_files,
                self.score_model,
                config.aposcore_config.checkpoint_path,
                config.aposcore_config.device,
                config=scoring_config,
                random_seed=config.random_seed,
            )

            # Convert scores to a format that the results processor can handle
            if isinstance(scored_poses, np.ndarray):
                logger.info(f"Received {len(scored_poses)} scores from get_mdn_score")

                # Count the total number of molecules across all SDF files
                total_molecules = 0
                for sdf_file in docked_ligands:
                    mols, _ = read_sdf_file(sdf_file, save_mols=True)
                    if mols is not None:
                        total_molecules += len(mols)

                logger.info(
                    f"Found a total of {total_molecules} molecules across {len(docked_ligands)} SDF files"
                )

                # Check if scores match the total number of molecules
                if len(scored_poses) != total_molecules:
                    logger.warning(
                        f"Mismatch between scores ({len(scored_poses)}) and molecules ({total_molecules})"
                    )

                # Create a list of (sdf_file, scores) pairs for processing
                # We'll let the results processor handle the matching of scores to molecules
                processed_scores = []
                score_index = 0
                for sdf_file in docked_ligands:
                    # Count molecules in this SDF file
                    mols, _ = read_sdf_file(sdf_file, save_mols=True)
                    if mols is None:
                        continue

                    num_mols = len(mols)
                    if score_index + num_mols <= len(scored_poses):
                        # Get scores for molecules in this file
                        file_scores = scored_poses[score_index : score_index + num_mols]
                        score_index += num_mols
                    else:
                        # If we don't have enough scores, use available ones and log warning
                        remaining = len(scored_poses) - score_index
                        if remaining > 0:
                            file_scores = scored_poses[score_index:]
                            score_index = len(scored_poses)
                        else:
                            # No scores left, set to empty list
                            file_scores = []
                        logger.warning(
                            f"Not enough scores for file {sdf_file}: needed {num_mols}, got {len(file_scores)}"
                        )

                    processed_scores.append((sdf_file, file_scores))

                scored_poses = processed_scores
                logger.info(
                    f"Prepared scored poses for processing: {len(processed_scores)} SDF files with scores"
                )

            # Step 4: Process results
            logger.info("Processing results...")
            final_poses = self.results_processor.process(
                scored_poses, config.output_dir
            )

            # Clean up temporary pocket files
            if config.screening_mode:
                # Track directories that might become empty after cleanup
                cleaned_dirs = set()
                
                # If we're in screening mode and using packing, clean up intermediate files
                if config.use_packing and not config.save_poses:
                    for protein_path in protein_list:
                        protein_id = os.path.basename(protein_path).split("_protein")[0]
                        # Find and remove all intermediate packed protein files
                        pack_files = [
                            os.path.join(root, f) for root, _, files in os.walk(config.output_dir)
                            for f in files if (f"_{protein_id}_pack_" in f or f"{protein_id}_pack_" in f) 
                            and not f.endswith("_best_aposcore_protein.pdb")  # Keep the best structure
                        ]
                        
                        for packed_file in pack_files:
                            try:
                                # Track directory for potential cleanup
                                cleaned_dirs.add(os.path.dirname(packed_file))
                                os.remove(packed_file)
                                logger.debug(f"Removed intermediate packed structure: {packed_file}")
                            except Exception as e:
                                logger.warning(f"Failed to remove packed structure {packed_file}: {str(e)}")
                
                # Clean up pocket files
                self.pocket_extractor.cleanup_temp_files()
                
                # Clean up empty directories
                for dir_path in cleaned_dirs:
                    try:
                        # Check if directory exists and is empty
                        if os.path.exists(dir_path) and os.path.isdir(dir_path) and not os.listdir(dir_path):
                            os.rmdir(dir_path)
                            logger.debug(f"Removed empty directory: {dir_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove directory {dir_path}: {str(e)}")

            logger.info(
                f"Docking completed successfully. Results saved to {config.output_dir}"
            )
            return final_poses

        except Exception as e:
            logger.error(f"Error during docking: {str(e)}")
            # Clean up GPU memory if available
            try:
                if torch.cuda.is_available() and config.pack_config.device.startswith(
                    "cuda"
                ):
                    from apodock.utils import ModelManager

                    ModelManager.cleanup_gpu_memory(config.pack_config.device)
                if (
                    torch.cuda.is_available()
                    and config.aposcore_config.device.startswith("cuda")
                ):
                    from apodock.utils import ModelManager

                    ModelManager.cleanup_gpu_memory(config.aposcore_config.device)
            except Exception as cleanup_error:
                logger.error(f"Error during GPU cleanup: {str(cleanup_error)}")

            raise
