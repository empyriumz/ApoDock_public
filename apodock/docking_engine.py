import os
import subprocess
from typing import List, Optional, Tuple

from apodock.config import DockingEngineConfig
from apodock.utils import ensure_dir, ApoDockError, logger


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
        # Extract IDs for output file naming
        ligand_name = os.path.basename(ligand).split(".")[0]
        protein_name = os.path.basename(protein).split(".")[0]
        ligand_id = ligand_name.split("_ligand")[0]
        protein_id = protein_name.split("_protein")[0]

        # Create protein-specific output directory
        protein_out_dir = os.path.join(out_dir, protein_id)
        ensure_dir(protein_out_dir)

        # Determine output file path based on settings
        if self.program == "smina.static":
            program_name = "smina"
        else:
            program_name = self.program

        if use_packing:
            output_path = os.path.join(
                protein_out_dir,
                f"{protein_id.split('.')[0]}_{program_name}_dock_{ligand_id}.sdf",
            )
        else:
            output_path = os.path.join(
                protein_out_dir, f"{program_name}_dock_{ligand_id}.sdf"
            )

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

    def dock_to_packed_proteins(
        self,
        ligand_list: List[str],
        cluster_packs_list: List[List[str]],
        ref_lig_list: List[str],
        out_dir: str,
    ) -> Tuple[List[str], List[str]]:
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
