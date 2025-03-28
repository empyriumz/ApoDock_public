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

    def _create_working_copy(
        self, file_path: str, work_dir: str, prefix: str = ""
    ) -> str:
        """
        Create a working copy of a file in the specified directory.

        Args:
            file_path: Path to the original file
            work_dir: Directory to create the copy in
            prefix: Optional prefix for the copied file name

        Returns:
            Path to the working copy
        """
        import shutil

        ensure_dir(work_dir)

        # Create a unique name for the working copy
        base_name = os.path.basename(file_path)
        if prefix:
            base_name = f"{prefix}_{base_name}"
        work_copy_path = os.path.join(work_dir, base_name)

        # Create the copy
        shutil.copy2(file_path, work_copy_path)
        logger.debug(f"Created working copy: {work_copy_path}")

        return work_copy_path

    def dock(
        self,
        ligand: str,
        protein: str,
        ref_lig: Optional[str] = None,
        out_dir: str = "./",
        use_packing: bool = True,
        base_filename: Optional[str] = None,
    ) -> str:
        """
        Dock a ligand to a protein using GNINA.
        Creates working copies of all input files to prevent modification of originals.

        Args:
            ligand: Path to the ligand file
            protein: Path to the protein file
            ref_lig: Path to the reference ligand (for defining the box), or None
            out_dir: Output directory for docking results
            use_packing: Whether packing was used (affects output file naming)
            base_filename: The original protein filename to use for directory organization

        Returns:
            Path to the output docked poses file
        """
        if not base_filename:
            raise ApoDockError(
                "base_filename must be provided for directory organization"
            )

        # Create protein-specific output directory using the full filename
        protein_out_dir = os.path.join(out_dir, base_filename)
        ensure_dir(protein_out_dir)

        # Create a working directory for input file copies
        work_dir = os.path.join(protein_out_dir, "work")
        ensure_dir(work_dir)

        try:
            # Create working copies of input files
            work_protein = self._create_working_copy(protein, work_dir, "protein")
            work_ligand = self._create_working_copy(ligand, work_dir, "ligand")
            work_ref_lig = (
                self._create_working_copy(ref_lig, work_dir, "ref") if ref_lig else None
            )

            # Extract protein name for output file naming (for packed structures)
            protein_name = os.path.basename(protein).split(".")[0]

            # Determine output file path
            if use_packing:
                output_path = os.path.join(protein_out_dir, f"{protein_name}_dock.sdf")
            else:
                output_path = os.path.join(protein_out_dir, f"{base_filename}_dock.sdf")

            # Create a log file path
            log_path = os.path.join(
                protein_out_dir, f"{os.path.basename(output_path).split('.')[0]}.log"
            )

            # Build and execute docking command using working copies
            if work_ref_lig:
                cmd = (
                    f"{self.gnina_path} --receptor {work_protein} --ligand {work_ligand} "
                    f"--autobox_ligand {work_ref_lig} --autobox_add {self.config.autobox_add} "
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
                    f"{self.gnina_path} -r {work_protein} -l {work_ligand} "
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
        finally:
            # Clean up working directory after docking
            import shutil

            if os.path.exists(work_dir):
                try:
                    shutil.rmtree(work_dir)
                    logger.debug(f"Cleaned up working directory: {work_dir}")
                except Exception as e:
                    logger.warning(
                        f"Failed to clean up working directory {work_dir}: {str(e)}"
                    )

    def dock_ligands_to_proteins(
        self,
        ligand_list: List[str],
        protein_list: List[str],
        ref_lig_list: List[str],
        out_dir: str,
        is_packed: bool = False,
        protein_filenames: Optional[List[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Dock a list of ligands to their corresponding protein structures.

        Args:
            ligand_list: List of paths to ligand files
            protein_list: List of paths to protein structures
            ref_lig_list: List of paths to reference ligand files
            out_dir: Output directory for docking results
            is_packed: Whether the proteins are results of packing
            protein_filenames: List of original protein filenames for directory organization

        Returns:
            Tuple containing:
                - List of paths to the docked pose files
                - List of paths to the corresponding protein structures used for each pose

        Raises:
            ApoDockError: If input lists have mismatched lengths or if required files are missing
        """
        # Input validation for non-packed case
        if not isinstance(protein_list[0], list):
            if len(ligand_list) != len(protein_list):
                raise ApoDockError(
                    f"Mismatched input lengths: {len(ligand_list)} ligands but {len(protein_list)} proteins"
                )
            if len(ref_lig_list) != len(protein_list):
                raise ApoDockError(
                    f"Mismatched input lengths: {len(ref_lig_list)} reference ligands but {len(protein_list)} proteins"
                )
            if protein_filenames and len(protein_filenames) != len(protein_list):
                raise ApoDockError(
                    f"Mismatched input lengths: {len(protein_filenames)} filenames but {len(protein_list)} proteins"
                )

        # Flatten packed protein list if needed
        if protein_list and isinstance(protein_list[0], list):
            flat_protein_list = []
            flat_ligand_list = []
            flat_ref_lig_list = []
            flat_filenames = []

            for i, packed_proteins in enumerate(protein_list):
                if i >= len(ligand_list) or i >= len(ref_lig_list):
                    raise ApoDockError(
                        f"Missing ligand or reference ligand for protein group {i}"
                    )

                for protein in packed_proteins:
                    flat_protein_list.append(protein)
                    flat_ligand_list.append(ligand_list[i])

                    if i >= len(ref_lig_list):
                        raise ApoDockError(
                            f"Missing reference ligand for protein group {i}"
                        )
                    flat_ref_lig_list.append(ref_lig_list[i])

                    if protein_filenames:
                        if i >= len(protein_filenames):
                            raise ApoDockError(
                                f"Missing filename for protein group {i}"
                            )
                        flat_filenames.append(protein_filenames[i])

            protein_list = flat_protein_list
            ligand_list = flat_ligand_list
            ref_lig_list = flat_ref_lig_list
            protein_filenames = flat_filenames if flat_filenames else None

        # Filter packed structures if needed
        if is_packed:
            if not any("_pack_" in os.path.basename(p) for p in protein_list):
                raise ApoDockError("No packed structures found when is_packed=True")

            filtered_items = [
                (protein, ligand, ref_lig, filename)
                for protein, ligand, ref_lig, filename in zip(
                    protein_list,
                    ligand_list,
                    ref_lig_list,
                    (
                        protein_filenames
                        if protein_filenames
                        else [None] * len(protein_list)
                    ),
                )
                if "_pack_" in os.path.basename(protein)
            ]

            if not filtered_items:
                raise ApoDockError("No valid packed structures found for docking")

            protein_list, ligand_list, ref_lig_list, protein_filenames = zip(
                *filtered_items
            )

        # Validate all files exist before starting docking
        for protein, ligand, ref_lig in zip(protein_list, ligand_list, ref_lig_list):
            if not os.path.exists(protein):
                raise ApoDockError(f"Protein file not found: {protein}")
            if not os.path.exists(ligand):
                raise ApoDockError(f"Ligand file not found: {ligand}")
            if not os.path.exists(ref_lig):
                raise ApoDockError(f"Reference ligand file not found: {ref_lig}")

        # Perform docking for each protein-ligand pair
        docked_poses = []
        corresponding_proteins = []

        for i, (protein, ligand, ref_lig) in enumerate(
            zip(protein_list, ligand_list, ref_lig_list)
        ):
            try:
                base_filename = protein_filenames[i] if protein_filenames else None
                docked_pose = self.dock(
                    ligand=ligand,
                    protein=protein,
                    ref_lig=ref_lig,
                    out_dir=out_dir,
                    use_packing=is_packed,
                    base_filename=base_filename,
                )
                docked_poses.append(docked_pose)
                corresponding_proteins.append(protein)
            except Exception as e:
                logger.error(f"Failed to dock {ligand} to {protein}: {str(e)}")
                raise ApoDockError(f"Docking failed: {str(e)}")

        return docked_poses, corresponding_proteins
