import os
from typing import List, Dict
import numpy as np
import torch
import tempfile

from apodock.config import PipelineConfig
from apodock.utils import ApoDockError, logger, set_random_seed
from apodock.pocket_extractor import PocketExtractor
from apodock.docking_engine import DockingEngine
from apodock.results_processor import ResultsProcessor

from apodock.Pack_sc.inference import sc_pack, PackingConfig
from apodock.Pack_sc.Packnn import Pack
from apodock.Aposcore.inference_dataset import (
    get_mdn_score,
    ScoringConfig,
)
from apodock.Aposcore.Aposcore import Aposcore
from Bio.PDB import PDBParser


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

    def validate_protein_structure(self, protein_files: List[str]) -> None:
        """
        Validate that protein files contain side chains when use_packing=False.

        This function checks if proteins have more atoms than would be expected
        for backbone-only structures. A backbone-only structure would typically have
        only N, CA, C, and O atoms per residue.

        Args:
            protein_files: List of paths to protein files to validate

        Raises:
            ApoDockError: If use_packing=False and proteins appear to be backbone-only
        """
        if self.config.use_packing:
            # Skip validation if packing is enabled
            return

        parser = PDBParser(QUIET=True)

        for protein_file in protein_files:
            try:
                # Parse the protein structure
                structure = parser.get_structure("protein", protein_file)

                # Count total residues and residues with side chains
                total_residues = 0
                residues_with_side_chains = 0

                for model in structure:
                    for chain in model:
                        for residue in chain:
                            total_residues += 1

                            # Check if residue has more atoms than just backbone
                            # Backbone atoms are typically N, CA, C, O
                            atom_names = set(atom.name for atom in residue)
                            backbone_atoms = {"N", "CA", "C", "O"}

                            # If there are atoms beyond backbone atoms, it has side chains
                            if len(atom_names - backbone_atoms) > 0:
                                residues_with_side_chains += 1

                # Calculate percentage of residues with side chains
                if total_residues == 0:
                    logger.warning(f"No residues found in protein file: {protein_file}")
                    continue

                side_chain_percentage = (
                    residues_with_side_chains / total_residues
                ) * 100

                # If less than 10% of residues have side chains, consider it a backbone-only structure
                if side_chain_percentage < 10:
                    protein_name = os.path.basename(protein_file)
                    raise ApoDockError(
                        f"Protein '{protein_name}' appears to be a backbone-only structure. "
                        f"When use_packing=False, proteins must include side chains. "
                        f"Either enable packing with use_packing=True or provide a complete protein structure."
                    )

            except Exception as e:
                if isinstance(e, ApoDockError):
                    # Re-raise ApoDockError exceptions
                    raise
                # Log other exceptions but continue validation
                logger.warning(
                    f"Error validating protein structure {protein_file}: {str(e)}"
                )

    def run_screening(
        self,
        ligand_list: List[str],
        protein_list: List[str],
        ref_lig_list: List[str],
        save_poses: bool = False,
        output_scores_file: str = None,
        rank_by: str = "aposcore",
    ) -> Dict[str, Dict[str, float]]:
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
        best_scores = {}  # Initialize return value
        temp_dir = None  # Track temporary directory if created

        try:
            # Validate protein structures if not using packing
            self.validate_protein_structure(protein_list)

            # Extract protein IDs for directory organization
            protein_ids = [
                os.path.basename(p).split("_protein")[0] for p in protein_list
            ]

            # Create output directory if it doesn't exist
            os.makedirs(config.output_dir, exist_ok=True)

            # Step 1: Extract pockets from proteins using reference ligands
            logger.info("Extracting pockets from proteins...")
            pocket_list = self.pocket_extractor.extract_pockets(
                protein_list, ref_lig_list, config.output_dir
            )

            # Step 2: Decide on strategy - packing or direct docking
            if config.use_packing:
                # With packing - first generate packed variants
                logger.info("Generating packed protein variants...")

                # Create a PackingConfig object from our existing config parameters
                packing_config = PackingConfig(
                    batch_size=config.pack_config.packing_batch_size,
                    number_of_packs_per_design=config.pack_config.packs_per_design,
                    temperature=config.pack_config.temperature,
                    n_recycle=3,  # Default value
                    resample=True,  # Default value
                    apo2holo=False,  # Explicitly set to False
                )

                # Determine working directory - use temp dir if not saving poses
                if not save_poses:
                    temp_dir = tempfile.mkdtemp(
                        prefix="packing_", dir=config.output_dir
                    )
                    working_dir = temp_dir
                    cleanup_intermediates = True
                else:
                    working_dir = config.output_dir
                    cleanup_intermediates = False

                # Run packing with appropriate directory settings
                cluster_packs_list = sc_pack(
                    ligand_list,
                    pocket_list,
                    protein_list,
                    self.pack_model,
                    config.pack_config.checkpoint_path,
                    config.pack_config.device,
                    working_dir,
                    config=packing_config,
                    ligandmpnn_path=config.pack_config.ligandmpnn_path,
                    num_clusters=config.pack_config.num_clusters,
                    random_seed=config.random_seed,
                    cleanup_intermediates=cleanup_intermediates,
                )

                # Step 3: Dock ligands to packed proteins
                logger.info("Docking ligands to packed proteins...")
                docking_results = self.docking_engine.dock_to_packed_proteins(
                    ligand_list, cluster_packs_list, ref_lig_list, working_dir
                )
                docked_ligands, scoring_pocket_files = docking_results

            else:
                # Direct docking (no packing)
                logger.info("Docking ligands to protein pockets...")
                docked_ligands = self.docking_engine.dock_to_pockets(
                    ligand_list, pocket_list, ref_lig_list, config.output_dir
                )
                scoring_pocket_files = pocket_list

                # For direct docking, ensure matching lengths between docked ligands and pockets
                if len(docked_ligands) != len(scoring_pocket_files):
                    logger.warning(
                        f"Mismatch in direct docking: {len(docked_ligands)} docked ligands but {len(scoring_pocket_files)} pockets"
                    )
                    # Use the shortest list length to avoid index errors
                    min_len = min(len(docked_ligands), len(scoring_pocket_files))
                    docked_ligands = docked_ligands[:min_len]
                    scoring_pocket_files = scoring_pocket_files[:min_len]

            # Step 4: Score docked poses - common for both paths
            logger.info("Scoring docked poses...")
            scoring_config = ScoringConfig(
                batch_size=64,  # Default batch size
                dis_threshold=5.0,  # Default distance threshold
                output_dir=config.output_dir,
            )

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

            # Step 5: Process scores and save best structures using results processor
            if isinstance(scores, np.ndarray):
                best_scores = self.results_processor.process_screening_results(
                    scores,
                    docked_ligands,
                    scoring_pocket_files,
                    protein_ids,
                    config.output_dir,
                    output_scores_file,
                    rank_by,
                    save_best_structures=True,  # Always save best structures
                )
            else:
                logger.error("Failed to obtain scores from the scoring model")

            return best_scores

        except Exception as e:
            logger.error(f"Error during screening: {str(e)}")
            # Clean up GPU memory if available
            try:
                if torch.cuda.is_available():
                    if config.pack_config.device.startswith("cuda"):
                        from apodock.utils import ModelManager

                        ModelManager.cleanup_gpu_memory(config.pack_config.device)
                    if config.aposcore_config.device.startswith("cuda"):
                        from apodock.utils import ModelManager

                        ModelManager.cleanup_gpu_memory(config.aposcore_config.device)
            except Exception as cleanup_error:
                logger.error(f"Error during GPU cleanup: {str(cleanup_error)}")
            raise
        finally:
            # Ensure cleanup happens regardless of success or failure
            if not save_poses:
                # Clean up pocket extractor temps
                self.pocket_extractor.cleanup_temp_files()

                # Clean up temp directory if we created one
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        import shutil

                        shutil.rmtree(temp_dir)
                        logger.debug(f"Removed temporary directory: {temp_dir}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to remove temporary directory {temp_dir}: {str(e)}"
                        )

                # Additional cleanup for intermediate packed files if using packing
                if config.use_packing:
                    self._cleanup_intermediate_files(
                        protein_list, config.output_dir, rank_by
                    )

    def _cleanup_intermediate_files(
        self, protein_list: List[str], output_dir: str, rank_by: str
    ) -> None:
        """
        Clean up intermediate packed protein files.

        Args:
            protein_list: List of paths to protein files
            output_dir: Output directory containing files to clean
            rank_by: Ranking criterion used to determine which files to keep
        """
        # Track directories that might become empty after cleanup
        cleaned_dirs = set()

        for protein_path in protein_list:
            protein_id = os.path.basename(protein_path).split("_protein")[0]
            # Find and remove all intermediate packed protein files
            pack_files = [
                os.path.join(root, f)
                for root, _, files in os.walk(output_dir)
                for f in files
                if (f"_{protein_id}_pack_" in f or f"{protein_id}_pack_" in f)
                and not f.endswith(
                    f"_best_{rank_by}_protein.pdb"
                )  # Keep the best structure
                and not f.endswith(
                    f"_best_{rank_by}_ligand.sdf"
                )  # Keep the best ligand
            ]

            for packed_file in pack_files:
                try:
                    # Track directory for potential cleanup
                    cleaned_dirs.add(os.path.dirname(packed_file))
                    os.remove(packed_file)
                    logger.debug(
                        f"Removed intermediate packed structure: {packed_file}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to remove packed structure {packed_file}: {str(e)}"
                    )

        # Clean up empty directories
        for dir_path in cleaned_dirs:
            try:
                # Check if directory exists and is empty
                if (
                    os.path.exists(dir_path)
                    and os.path.isdir(dir_path)
                    and not os.listdir(dir_path)
                ):
                    os.rmdir(dir_path)
                    logger.debug(f"Removed empty directory: {dir_path}")
            except Exception as e:
                logger.warning(f"Failed to remove directory {dir_path}: {str(e)}")
