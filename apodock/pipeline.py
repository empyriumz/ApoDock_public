import os
import subprocess
from typing import List, Optional
from rdkit import Chem

from apodock.config import PipelineConfig, DockingEngineConfig
from apodock.utils import ensure_dir, ApoDockError, logger
from apodock.Pack_sc.inference import sc_pack
from apodock.Pack_sc.Packnn import Pack
from apodock.Aposcore.inference_dataset import get_mdn_score, read_sdf_file
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

    def __init__(self, config: DockingEngineConfig):
        """
        Initialize the docking engine.

        Args:
            config: Configuration for the docking engine
        """
        self.config = config
        self.program = config.program
        self.gnina_path = config.gnina_path

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
                f"--exhaustiveness {self.config.exhaustiveness}"
            )

        try:
            logger.info(f"Running docking command: {cmd}")
            subprocess.run(cmd, shell=True, check=True)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Docking failed: {str(e)}")
            raise ApoDockError(f"Docking failed: {str(e)}")


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
        if not (len(mol_list) == len(mol_name_list) == len(scores)):
            logger.error(
                f"Mismatched counts: {len(mol_list)} molecules, {len(mol_name_list)} names, {len(scores)} scores"
            )
            return None

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


class DockingPipeline:
    """Main docking pipeline class that orchestrates the docking process."""

    def __init__(self, config: PipelineConfig):
        """
        Initialize the docking pipeline.

        Args:
            config: Configuration for the docking pipeline
        """
        self.config = config
        self.pocket_extractor = PocketExtractor(config.pocket_distance)
        self.docking_engine = DockingEngine(config.docking_config)
        self.results_processor = ResultsProcessor(config.top_k)

        logger.info("Initializing machine learning models...")

        # Initialize models only if we're going to use them
        if config.use_packing:
            logger.info(f"Loading Pack model from {config.pack_config.checkpoint_path}")
            self.pack_model = Pack(recycle_strategy="sample")

        logger.info(
            f"Loading Aposcore model from {config.aposcore_config.checkpoint_path}"
        )
        self.score_model = Aposcore(
            35,
            hidden_dim=config.aposcore_config.hidden_dim,
            num_heads=config.aposcore_config.num_heads,
            dropout=config.aposcore_config.dropout,
            crossAttention=config.aposcore_config.cross_attention,
            atten_active_fuc=config.aposcore_config.attention_activation,
            num_layers=config.aposcore_config.num_layers,
            interact_type=config.aposcore_config.interaction_type,
        )

    def run(
        self, ligand_list: List[str], protein_list: List[str], ref_lig_list: List[str]
    ) -> List[str]:
        """
        Run the complete docking pipeline.

        Args:
            ligand_list: List of ligand file paths
            protein_list: List of protein file paths
            ref_lig_list: List of reference ligand file paths

        Returns:
            List of paths to the final ranked docking result files
        """
        config = self.config
        ensure_dir(config.output_dir)

        # Step 1: Extract protein pockets
        logger.info("Extracting protein pockets...")
        pocket_list = self.pocket_extractor.extract_pockets(
            protein_list, ref_lig_list, config.output_dir
        )

        result_files = []
        ids_list = [os.path.basename(i).split("_ligand")[0] for i in ligand_list]

        # Decide on strategy: packing or direct docking
        if config.use_packing:
            # Step 2a: With packing - do side chain packing first
            logger.info("Running side chain packing...")
            cluster_packs_list = sc_pack(
                ligand_list,
                pocket_list,
                protein_list,
                self.pack_model,
                config.pack_config.checkpoint_path,
                config.pack_config.device,
                config.pack_config.packing_batch_size,
                config.pack_config.packs_per_design,
                config.output_dir,
                config.pack_config.temperature,
                config.pack_config.ligandmpnn_path,
                apo2holo=False,
                num_clusters=config.pack_config.num_clusters,
            )

            # Step 3a: Dock against packed structures
            for i, protein_id in enumerate(ids_list):
                logger.info(f"Docking for {protein_id}...")
                protein_dir = os.path.dirname(pocket_list[i])

                # Get packed pockets for this protein
                packed_pockets = next(
                    (p for p in cluster_packs_list if any(protein_id in j for j in p)),
                    None,
                )

                if not packed_pockets:
                    logger.warning(
                        f"No packed pockets found for {protein_id}, skipping"
                    )
                    continue

                # Clean up old docked files
                old_files = [
                    os.path.join(protein_dir, f)
                    for f in os.listdir(protein_dir)
                    if f.endswith(".sdf") and "_pack_" in f
                ]
                for old_file in old_files:
                    os.remove(old_file)

                # Dock against each packed pocket
                out_sdfs = []
                for packed_pdb in packed_pockets:
                    out_sdf = self.docking_engine.dock(
                        ligand_list[i],
                        packed_pdb,
                        ref_lig_list[i],
                        protein_dir,
                        use_packing=True,
                    )
                    out_sdfs.append(out_sdf)

                # Score docked poses
                logger.info(f"Scoring docked poses for {protein_id}...")
                scores = get_mdn_score(
                    out_sdfs,
                    packed_pockets,
                    self.score_model,
                    config.aposcore_config.checkpoint_path,
                    config.aposcore_config.device,
                    dis_threshold=5.0,
                )

                # Rank and process results
                result_file = self.results_processor.rank_results(
                    out_sdfs,
                    scores,
                    use_packing=True,
                    docking_program=config.docking_config.program,
                )
                if result_file:
                    result_files.append(result_file)
        else:
            # Step 2b: Without packing - direct docking
            for i, protein_id in enumerate(ids_list):
                logger.info(f"Docking for {protein_id} without packing...")
                protein_dir = os.path.dirname(pocket_list[i])

                # Dock against the pocket directly
                out_sdf = self.docking_engine.dock(
                    ligand_list[i],
                    pocket_list[i],
                    ref_lig_list[i],
                    protein_dir,
                    use_packing=False,
                )

                # Score the docked poses
                logger.info(f"Scoring docked poses for {protein_id}...")
                scores = get_mdn_score(
                    [out_sdf],
                    [pocket_list[i]],
                    self.score_model,
                    config.aposcore_config.checkpoint_path,
                    config.aposcore_config.device,
                    dis_threshold=5.0,
                )

                # Rank and process results
                result_file = self.results_processor.rank_results(
                    [out_sdf],
                    scores,
                    use_packing=False,
                    docking_program=config.docking_config.program,
                )
                if result_file:
                    result_files.append(result_file)

        logger.info("Docking pipeline completed successfully")
        return result_files
