import os
from typing import List
from apodock.utils import ensure_dir, logger


class PocketExtractor:
    """Extract pockets from protein structures based on reference ligands or specified centers."""

    def __init__(self, distance: float = 10.0):
        """
        Initialize the pocket extractor.

        Args:
            distance: Distance in Angstroms to define the pocket around the reference ligand
        """
        self.distance = distance
        self.temp_files = []  # Track temporary files for cleanup

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

        # Add to temporary files list for potential cleanup
        self.temp_files.append(pocket_path)

        # Also check if a "Pocket_clean_XX.pdb" file might be created
        protein_id = os.path.basename(protein).split("_protein")[0]
        clean_pocket_path = os.path.join(out_dir, f"Pocket_clean_{protein_id}.pdb")
        if os.path.exists(clean_pocket_path):
            self.temp_files.append(clean_pocket_path)

        return pocket_path

    def cleanup_temp_files(self):
        """
        Remove temporary pocket files that are no longer needed.
        Also removes empty directories that might be left after cleaning up files.
        """
        # Track directories that might become empty
        cleaned_dirs = set()

        for file_path in self.temp_files:
            if os.path.exists(file_path):
                try:
                    # Track the directory containing this file
                    cleaned_dirs.add(os.path.dirname(file_path))
                    os.remove(file_path)
                    logger.debug(f"Removed temporary pocket file: {file_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to remove temporary file {file_path}: {str(e)}"
                    )

        # Clear the list after cleanup
        self.temp_files = []

        # Remove empty directories
        for dir_path in cleaned_dirs:
            try:
                # Check if directory exists and is empty
                if (
                    os.path.exists(dir_path)
                    and os.path.isdir(dir_path)
                    and not os.listdir(dir_path)
                ):
                    os.rmdir(dir_path)
                    logger.debug(f"Removed empty pocket directory: {dir_path}")
            except Exception as e:
                logger.warning(f"Failed to remove directory {dir_path}: {str(e)}")

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
