import os
import subprocess
from typing import List, Optional
from rdkit import Chem
import numpy as np
import torch

from apodock.config import PipelineConfig, DockingEngineConfig
from apodock.utils import ensure_dir, ApoDockError, logger, set_random_seed
from apodock.Pack_sc.inference import sc_pack
from apodock.Pack_sc.Packnn import Pack
from apodock.Aposcore.inference_dataset import (
    get_mdn_score,
    read_sdf_file,
    ScoringConfig,
)
from apodock.Aposcore.Aposcore import Aposcore


class PocketExtractor:
    """Extract pockets from protein structures based on reference ligands or specified centers."""

    def __init__(self, distance: float = 10.0):
        """
        Initialize the pocket extractor.

        Args:
            distance: Distance in Angstroms to define the pocket around the reference ligand
        """
        self.distance = distance

    def extract_pocket(self, protein: str, ref_ligand: str, out_dir: str) -> str:
        """
        Extract a pocket from a protein based on a reference ligand.

        Args:
            protein: Path to the protein file
            ref_ligand: Path to the reference ligand file
            out_dir: Output directory for the pocket file

        Returns:
            Path to the generated pocket file
        """
        import pymol

        ensure_dir(out_dir)

        # Extract the pocket using PyMOL
        pymol.cmd.load(protein, "protein")
        pymol.cmd.remove("resn HOH")
        pymol.cmd.remove("not polymer.protein")
        pymol.cmd.load(ref_ligand, "ligand")
        pymol.cmd.remove("hydrogens")
        pymol.cmd.select("Pocket", f"byres ligand around {self.distance}")

        pocket_path = os.path.join(out_dir, f"Pocket_{self.distance}A.pdb")
        pymol.cmd.save(pocket_path, "Pocket")
        pymol.cmd.delete("all")

        return pocket_path

    def extract_pockets(
        self, protein_list: List[str], ref_lig_list: List[str], out_dir: str
    ) -> List[str]:
        """
        Extract pockets from a list of proteins based on reference ligands.

        Args:
            protein_list: List of protein file paths
            ref_lig_list: List of reference ligand file paths
            out_dir: Base output directory

        Returns:
            List of paths to the generated pocket files
        """
        out_pocket_paths = []

        for protein, ref_lig in zip(protein_list, ref_lig_list):
            protein_id = os.path.basename(protein).split("_protein")[0]
            protein_out_dir = os.path.join(out_dir, protein_id)

            ensure_dir(protein_out_dir)
            pocket_path = self.extract_pocket(protein, ref_lig, protein_out_dir)
            out_pocket_paths.append(pocket_path)

        return out_pocket_paths


class DockingEngine:
    """Interface to external docking programs like gnina or smina."""

    def __init__(self, config: DockingEngineConfig, random_seed: int = 42):
        """
        Initialize the docking engine.

        Args:
            config: Configuration for the docking engine
            random_seed: Random seed for reproducibility (default: 42)
        """
        self.config = config
        self.program = config.program
        self.gnina_path = config.gnina_path
        self.random_seed = random_seed

        # Log whether box center and size are defined in the config
        if config.box_center:
            logger.info(f"Using box center from config: {config.box_center}")
        else:
            logger.info(
                "Box center not defined in config, will use reference ligand for box definition"
            )

        if config.box_size:
            logger.info(f"Using box size from config: {config.box_size}")
        else:
            logger.info(
                "Box size not defined in config, will use reference ligand for box definition"
            )

        logger.info(f"Docking engine initialized with random seed: {random_seed}")

    def dock(
        self,
        ligand: str,
        protein: str,
        ref_lig: Optional[str] = None,
        out_dir: str = "./",
        use_packing: bool = True,
    ) -> str:
        """
        Dock a ligand to a protein using the configured docking program.

        Args:
            ligand: Path to the ligand file
            protein: Path to the protein file
            ref_lig: Path to the reference ligand (for defining the box), or None
            out_dir: Output directory for docking results
            use_packing: Whether packing was used (affects output file naming)

        Returns:
            Path to the output docked poses file
        """
        ensure_dir(out_dir)

        # Extract IDs for output file naming
        ligand_name = os.path.basename(ligand).split(".")[0]
        protein_name = os.path.basename(protein).split(".")[0]
        ligand_id = ligand_name.split("_ligand")[0]
        protein_id = protein_name.split("_protein")[0]

        # Determine output file path based on settings
        if self.program == "smina.static":
            program_name = "smina"
        else:
            program_name = self.program

        if use_packing:
            output_path = os.path.join(
                out_dir,
                f"{protein_id.split('.')[0]}_{program_name}_dock_{ligand_id}.sdf",
            )
        else:
            output_path = os.path.join(out_dir, f"{program_name}_dock_{ligand_id}.sdf")

        # Build and execute docking command
        if ref_lig:
            cmd = (
                f"{self.gnina_path} --receptor {protein} --ligand {ligand} "
                f"--autobox_ligand {ref_lig} --autobox_add {self.config.autobox_add} "
                f"--num_modes {self.config.num_modes} --exhaustiveness {self.config.exhaustiveness} "
                f"--seed {self.random_seed} "
                f"--out {output_path}"
            )
        else:
            if not (self.config.box_center and self.config.box_size):
                raise ApoDockError(
                    "Box center and size must be provided when not using a reference ligand"
                )

            cmd = (
                f"{self.gnina_path} -r {protein} -l {ligand} "
                f"--center_x {self.config.box_center[0]} --center_y {self.config.box_center[1]} "
                f"--center_z {self.config.box_center[2]} --size_x {self.config.box_size[0]} "
                f"--size_y {self.config.box_size[1]} --size_z {self.config.box_size[2]} "
                f"--num_modes {self.config.num_modes} -o {output_path} "
                f"--exhaustiveness {self.config.exhaustiveness} "
                f"--seed {self.random_seed}"
            )

        try:
            logger.info(f"Running docking command: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Docking failed: {str(e)}")
            raise ApoDockError(f"Docking failed: {str(e)}")

    def dock_to_packed_proteins(
        self,
        ligand_list: List[str],
        cluster_packs_list: List[List[str]],
        ref_lig_list: List[str],
        out_dir: str,
    ) -> tuple:
        """
        Dock a list of ligands to their corresponding packed protein structures.

        Args:
            ligand_list: List of paths to ligand files
            cluster_packs_list: List of lists of packed protein structures
            ref_lig_list: List of paths to reference ligand files
            out_dir: Output directory for docking results

        Returns:
            Tuple containing:
                - List of paths to the docked pose files
                - List of paths to the corresponding packed protein structures used for each pose
        """
        docked_poses = []
        corresponding_packed_proteins = (
            []
        )  # Track which packed protein was used for each pose

        for i, ligand in enumerate(ligand_list):
            # Get the reference ligand for this ligand/protein pair
            ref_lig = ref_lig_list[i] if i < len(ref_lig_list) else None

            ligand_poses = []
            ligand_proteins = []  # Store packed proteins used for this ligand
            # Dock the ligand to each packed protein in the corresponding cluster
            for packed_protein in cluster_packs_list[i]:
                try:
                    # Use the original reference ligand from ref_lig_list instead of searching in the directory
                    # If no reference ligand provided, try to find one in the protein directory as fallback
                    if ref_lig is None or not os.path.exists(ref_lig):
                        # Fallback to searching in the protein directory
                        protein_dir = os.path.dirname(packed_protein)
                        ref_lig_candidates = [
                            f
                            for f in os.listdir(protein_dir)
                            if f.endswith(".mol2") or f.endswith(".sdf")
                        ]

                        local_ref_lig = None
                        if ref_lig_candidates:
                            local_ref_lig = os.path.join(
                                protein_dir, ref_lig_candidates[0]
                            )
                            logger.info(
                                f"Found local reference ligand: {local_ref_lig}"
                            )
                    else:
                        local_ref_lig = ref_lig
                        logger.info(f"Using provided reference ligand: {local_ref_lig}")

                    # Dock using the packed protein
                    docked_pose = self.dock(
                        ligand=ligand,
                        protein=packed_protein,
                        ref_lig=local_ref_lig,
                        out_dir=out_dir,
                        use_packing=True,
                    )
                    ligand_poses.append(docked_pose)
                    ligand_proteins.append(
                        packed_protein
                    )  # Store the packed protein used
                except Exception as e:
                    logger.warning(
                        f"Failed to dock {ligand} to {packed_protein}: {str(e)}"
                    )

            if ligand_poses:
                docked_poses.extend(ligand_poses)
                corresponding_packed_proteins.extend(ligand_proteins)
            else:
                logger.error(f"No successful docking for ligand {ligand}")

        return docked_poses, corresponding_packed_proteins

    def dock_to_pockets(
        self,
        ligand_list: List[str],
        pocket_list: List[str],
        ref_lig_list: List[str],
        out_dir: str,
    ) -> List[str]:
        """
        Dock a list of ligands to their corresponding protein pockets.

        Args:
            ligand_list: List of paths to ligand files
            pocket_list: List of paths to protein pocket files
            ref_lig_list: List of paths to reference ligand files
            out_dir: Output directory for docking results

        Returns:
            List of paths to the docked pose files
        """
        docked_poses = []

        for i, (ligand, pocket) in enumerate(zip(ligand_list, pocket_list)):
            try:
                # Use the original reference ligand if available
                ref_lig = ref_lig_list[i] if i < len(ref_lig_list) else None

                # If no reference ligand provided or it doesn't exist, look in the pocket directory as fallback
                if ref_lig is None or not os.path.exists(ref_lig):
                    # Fallback to searching in the pocket directory
                    pocket_dir = os.path.dirname(pocket)
                    ref_lig_candidates = [
                        f
                        for f in os.listdir(pocket_dir)
                        if f.endswith(".mol2") or f.endswith(".sdf")
                    ]

                    if ref_lig_candidates:
                        ref_lig = os.path.join(pocket_dir, ref_lig_candidates[0])
                        logger.info(f"Found local reference ligand: {ref_lig}")
                else:
                    logger.info(f"Using provided reference ligand: {ref_lig}")

                # Dock the ligand to the pocket
                docked_pose = self.dock(
                    ligand=ligand,
                    protein=pocket,
                    ref_lig=ref_lig,
                    out_dir=out_dir,
                    use_packing=False,
                )
                docked_poses.append(docked_pose)
            except Exception as e:
                logger.error(f"Failed to dock {ligand} to {pocket}: {str(e)}")

        return docked_poses


class ResultsProcessor:
    """Process and rank docking results."""

    def __init__(self, top_k: int = 10):
        """
        Initialize the results processor.

        Args:
            top_k: Number of top poses to keep in the final output
        """
        self.top_k = top_k

    def rank_results(
        self,
        docked_sdfs: List[str],
        scores: List[float],
        use_packing: bool = True,
        docking_program: str = "smina",
    ) -> Optional[str]:
        """
        Rank docking results based on scores and write top poses to a file.

        Args:
            docked_sdfs: List of paths to docked pose SDF files
            scores: List of scores for each pose
            use_packing: Whether packing was used
            docking_program: Name of the docking program used

        Returns:
            Path to the ranked poses file, or None if processing failed
        """
        if docking_program == "smina.static":
            docking_program = "smina"

        # Read all molecules from SDF files
        mol_list = []
        mol_name_list = []
        for sdf in docked_sdfs:
            mols, mol_names = read_sdf_file(sdf, save_mols=True)
            if mols is None:
                continue
            mol_list.extend(mols)
            mol_name_list.extend(mol_names)

        if len(mol_list) == 0:
            logger.error(
                "No molecules were read from the SDF files or passed the RDKit sanitization"
            )
            return None

        # Check that counts match
        if len(mol_list) != len(mol_name_list):
            logger.error(
                f"Mismatched molecule and name counts: {len(mol_list)} molecules, {len(mol_name_list)} names"
            )
            return None

        # Handle potential score count mismatch
        if len(mol_list) != len(scores):
            logger.warning(
                f"Mismatched molecule and score counts: {len(mol_list)} molecules, {len(scores)} scores"
            )

            # Adjust scores list if needed
            if len(scores) > len(mol_list):
                # If we have more scores than molecules, truncate the scores list
                logger.warning(
                    f"Truncating scores list from {len(scores)} to {len(mol_list)}"
                )
                scores = scores[: len(mol_list)]
            else:
                # If we have fewer scores than molecules, we'll duplicate the last score
                # This is a fallback approach to allow processing to continue
                logger.warning(
                    f"Extending scores list from {len(scores)} to {len(mol_list)} by duplicating last score"
                )
                last_score = scores[-1] if scores else 0.0
                scores.extend([last_score] * (len(mol_list) - len(scores)))

        # Sort molecules by scores
        sorted_pairs = sorted(
            zip(scores, mol_list, mol_name_list), key=lambda pair: pair[0], reverse=True
        )

        sorted_mol_list = [mol for _, mol, _ in sorted_pairs]
        sorted_mol_name_list = [name for _, _, name in sorted_pairs]

        # Take top-k poses
        top_k_mol_list = sorted_mol_list[: self.top_k]
        top_k_mol_name_list = sorted_mol_name_list[: self.top_k]

        # Write to output file
        if use_packing:
            top_k_sdf_filename = (
                f"Packed_top_{self.top_k}_{docking_program}_rescore_poses.sdf"
            )
        else:
            top_k_sdf_filename = f"Top_{self.top_k}_{docking_program}_rescore_poses.sdf"

        top_k_sdf = os.path.join(os.path.dirname(docked_sdfs[0]), top_k_sdf_filename)

        with Chem.SDWriter(top_k_sdf) as w:
            for i, mol in enumerate(top_k_mol_list):
                mol.SetProp("_Name", top_k_mol_name_list[i])
                w.write(mol)

        logger.info(f"Wrote top {self.top_k} poses to {top_k_sdf}")
        return top_k_sdf

    def process(self, scored_poses: List[tuple], output_dir: str) -> List[str]:
        """
        Process scored poses and rank them.

        Args:
            scored_poses: List of tuples containing (docked_sdf_path, scores) or directly an array of scores
            output_dir: Output directory for processed results

        Returns:
            List of paths to the ranked pose files
        """
        result_files = []

        # Handle direct scores array from get_mdn_score
        if isinstance(scored_poses, np.ndarray):
            logger.warning(
                "Received raw scores array without file paths, cannot process results"
            )
            return result_files

        # Check if we have paired data (docked_sdf, scores) or just a list of files followed by separate scores
        if all(isinstance(item, tuple) and len(item) == 2 for item in scored_poses):
            # We have paired data - extract files and scores separately
            docked_sdfs = [item[0] for item in scored_poses]
            all_scores = []
            for _, score in scored_poses:
                if isinstance(score, (list, np.ndarray)):
                    all_scores.extend(
                        score if isinstance(score, list) else score.tolist()
                    )
                else:
                    all_scores.append(score)

            # Group the SDF files by directory (to handle different ligand/protein pairs)
            grouped_sdfs = {}
            for sdf_file in docked_sdfs:
                dir_key = os.path.dirname(sdf_file)
                if dir_key not in grouped_sdfs:
                    grouped_sdfs[dir_key] = []
                grouped_sdfs[dir_key].append(sdf_file)

            # Process each group
            for dir_key, sdf_files in grouped_sdfs.items():
                # Determine if packing was used based on file name pattern
                use_packing = any(
                    "Packed" in os.path.basename(f) or "_pack_" in os.path.basename(f)
                    for f in sdf_files
                )

                # Get the docking program name from the file path
                docking_program = "smina"  # Default
                for f in sdf_files:
                    if "gnina" in f:
                        docking_program = "gnina"
                        break

                # Count total molecules to calculate how many scores we need
                total_mols = 0
                mol_counts = []
                for sdf in sdf_files:
                    mols, _ = read_sdf_file(sdf, save_mols=True)
                    if mols is None:
                        mol_counts.append(0)
                        continue
                    mol_counts.append(len(mols))
                    total_mols += len(mols)

                # If total molecules doesn't match available scores, log warning and continue
                if total_mols > len(all_scores):
                    logger.warning(
                        f"Not enough scores ({len(all_scores)}) for all molecules ({total_mols})"
                    )
                    continue

                # Take the appropriate number of scores for this group's molecules
                group_scores = all_scores[:total_mols]
                all_scores = all_scores[total_mols:]  # Remove used scores

                # Rank the results
                ranked_file = self.rank_results(
                    sdf_files, group_scores, use_packing, docking_program
                )

                if ranked_file:
                    result_files.append(ranked_file)
        else:
            # Legacy format processing - keep for backward compatibility
            grouped_poses = {}
            for pose_info in scored_poses:
                if len(pose_info) != 2:
                    logger.warning(f"Unexpected pose info format: {pose_info}")
                    continue

                docked_sdf, scores = pose_info
                # Use the directory as a key to group related poses
                dir_key = os.path.dirname(docked_sdf)
                if dir_key not in grouped_poses:
                    grouped_poses[dir_key] = []
                grouped_poses[dir_key].append((docked_sdf, scores))

            # Process each group of poses
            for dir_key, pose_group in grouped_poses.items():
                sdf_files = [p[0] for p in pose_group]
                # Flatten the scores if they're lists
                all_scores = []
                for _, score_list in pose_group:
                    if isinstance(score_list, list):
                        all_scores.extend(score_list)
                    elif isinstance(score_list, (float, int)):
                        all_scores.append(score_list)
                    elif isinstance(score_list, np.ndarray):
                        all_scores.extend(score_list.tolist())
                    else:
                        logger.warning(f"Unexpected score type: {type(score_list)}")

                # Determine if packing was used based on file name pattern
                use_packing = any(
                    "Packed" in os.path.basename(f) or "_pack_" in os.path.basename(f)
                    for f in sdf_files
                )

                # Get the docking program name from the file path
                docking_program = "smina"  # Default
                for f in sdf_files:
                    if "gnina" in f:
                        docking_program = "gnina"
                        break

                # Rank the results
                ranked_file = self.rank_results(
                    sdf_files, all_scores, use_packing, docking_program
                )

                if ranked_file:
                    result_files.append(ranked_file)

        return result_files


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
