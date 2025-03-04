import os
import subprocess
from typing import List, Optional, Tuple

from apodock.config import DockingEngineConfig
from apodock.utils import ensure_dir, logger, ApoDockError


class DockingEngine:
    """Interface to external docking program (GNINA)."""

    def __init__(self, config: DockingEngineConfig, random_seed: int = 42):
        """
        Initialize the docking engine.

        Args:
            config: Configuration for the docking engine
            random_seed: Random seed for reproducibility (default: 42)
        """
        self.config = config
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

        logger.info(f"GNINA docking engine initialized with random seed: {random_seed}")

    def dock(
        self,
        ligand: str,
        protein: str,
        ref_lig: Optional[str] = None,
        out_dir: str = "./",
        use_packing: bool = True,
    ) -> str:
        """
        Dock a ligand to a protein using GNINA.

        Args:
            ligand: Path to the ligand file
            protein: Path to the protein file
            ref_lig: Path to the reference ligand (for defining the box), or None
            out_dir: Output directory for docking results
            use_packing: Whether packing was used (affects output file naming)

        Returns:
            Path to the output docked poses file
        """
        # Extract IDs for output file naming
        ligand_name = os.path.basename(ligand).split(".")[0]
        protein_name = os.path.basename(protein).split(".")[0]
        ligand_id = ligand_name.split("_ligand")[0]
        protein_id = protein_name.split("_protein")[0]

        # Create protein-specific output directory
        protein_out_dir = os.path.join(out_dir, protein_id)
        ensure_dir(protein_out_dir)

        # Determine output file path based on packing
        if use_packing:
            output_path = os.path.join(
                protein_out_dir,
                f"{protein_id.split('.')[0]}_gnina_dock_{ligand_id}.sdf",
            )
        else:
            output_path = os.path.join(protein_out_dir, f"gnina_dock_{ligand_id}.sdf")

        # Create a log file path to save the docking output
        log_path = os.path.join(
            protein_out_dir, f"{os.path.basename(output_path).split('.')[0]}.log"
        )

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
            # Run the command and capture the output
            with open(log_path, "w") as log_file:
                process = subprocess.run(
                    cmd,
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                # Write the output to the log file
                log_file.write(process.stdout)

            logger.info(
                f"Docking completed. Output saved to {output_path}, log saved to {log_path}"
            )
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Docking failed: {str(e)}")
            # If there was output, save it to the log file for debugging
            if hasattr(e, "output") and e.output:
                with open(log_path, "w") as log_file:
                    log_file.write(e.output)
            raise ApoDockError(f"Docking failed: {str(e)}")

    def dock_ligands_to_proteins(
        self,
        ligand_list: List[str],
        protein_list: List[str],
        ref_lig_list: List[str],
        out_dir: str,
        is_packed: bool = False,
    ) -> Tuple[List[str], List[str]]:
        """
        Dock a list of ligands to their corresponding protein structures.

        This method handles both packed and original protein structures.

        Args:
            ligand_list: List of paths to ligand files
            protein_list: List of paths to protein structures
                (can be flattened list of packed variants or list of original pockets)
            ref_lig_list: List of paths to reference ligand files
            out_dir: Output directory for docking results
            is_packed: Whether the proteins are results of packing (affects naming)

        Returns:
            Tuple containing:
                - List of paths to the docked pose files
                - List of paths to the corresponding protein structures used for each pose
        """
        docked_poses = []
        corresponding_proteins = []  # Track which protein was used for each pose

        # Create a flat list if protein_list is a list of lists (from packed proteins)
        if protein_list and isinstance(protein_list[0], list):
            # Flatten the list of packed proteins
            flat_protein_list = [p for sublist in protein_list for p in sublist]
            # Create matching ligand and reference ligand lists
            flat_ligand_list = []
            flat_ref_lig_list = []

            for i, packed_proteins in enumerate(protein_list):
                for _ in packed_proteins:
                    flat_ligand_list.append(ligand_list[i])
                    flat_ref_lig_list.append(
                        ref_lig_list[i] if i < len(ref_lig_list) else None
                    )

            protein_list = flat_protein_list
            ligand_list = flat_ligand_list
            ref_lig_list = flat_ref_lig_list

        # Filter out non-packed structures if is_packed is True
        if is_packed:
            filtered_proteins = []
            filtered_ligands = []
            filtered_ref_ligs = []
            for i, protein in enumerate(protein_list):
                if "_pack_" in os.path.basename(protein):
                    filtered_proteins.append(protein)
                    if i < len(ligand_list):
                        filtered_ligands.append(ligand_list[i])
                    if i < len(ref_lig_list):
                        filtered_ref_ligs.append(ref_lig_list[i])
            protein_list = filtered_proteins
            ligand_list = filtered_ligands
            ref_lig_list = filtered_ref_ligs

        # Ensure protein and ligand lists have same length
        if len(protein_list) != len(ligand_list):
            logger.warning(
                f"Mismatch in lengths: {len(ligand_list)} ligands but {len(protein_list)} proteins. "
                f"Using the shorter list."
            )
            min_len = min(len(ligand_list), len(protein_list))
            protein_list = protein_list[:min_len]
            ligand_list = ligand_list[:min_len]
            if len(ref_lig_list) > min_len:
                ref_lig_list = ref_lig_list[:min_len]

        # Perform docking for each protein-ligand pair
        for i, (ligand, protein) in enumerate(zip(ligand_list, protein_list)):
            try:
                # Get the reference ligand for this pair
                ref_lig = ref_lig_list[i] if i < len(ref_lig_list) else None

                # Ensure a valid reference ligand is provided
                if ref_lig is None or not os.path.exists(ref_lig):
                    raise ApoDockError(
                        f"Missing or invalid reference ligand for docking ligand {ligand} with protein {protein}. "
                        f"Reference ligand is required to define the binding site."
                    )

                # Dock the ligand to the protein
                docked_pose = self.dock(
                    ligand=ligand,
                    protein=protein,
                    ref_lig=ref_lig,
                    out_dir=out_dir,
                    use_packing=is_packed,
                )
                docked_poses.append(docked_pose)
                corresponding_proteins.append(protein)
            except Exception as e:
                logger.warning(f"Failed to dock {ligand} to {protein}: {str(e)}")

        return docked_poses, corresponding_proteins
