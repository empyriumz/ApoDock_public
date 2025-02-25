import os
import logging
from typing import List, Tuple, Optional
import pandas as pd
from rdkit import Chem

# Set up logging
logger = logging.getLogger("apodock")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ApoDockError(Exception):
    """Base exception for ApoDock errors."""

    pass


def ensure_dir(directory: str) -> str:
    """
    Ensure that a directory exists, create it if it doesn't.

    Args:
        directory: Path to the directory to check/create

    Returns:
        The path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory


def get_data_from_csv(csv_file: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Read CSV file containing the ligand, protein, and reference ligand paths.

    Args:
        csv_file: Path to the CSV file with columns 'ligand', 'protein', and 'ref_lig'

    Returns:
        Tuple containing lists of ligand, protein, and reference ligand paths
    """
    try:
        data = pd.read_csv(csv_file)
        ligand_list = data["ligand"].tolist()
        protein_list = data["protein"].tolist()
        ref_lig_list = data["ref_lig"].tolist()

        if not (len(ligand_list) == len(protein_list) == len(ref_lig_list)):
            raise ApoDockError(
                "Mismatched item counts in CSV: ligands, proteins, and reference ligands must have the same count"
            )

        return ligand_list, protein_list, ref_lig_list
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_file}: {str(e)}")
        raise ApoDockError(f"Failed to process CSV file: {str(e)}") from e


def read_molecule(mol_path: str) -> Optional[Chem.Mol]:
    """
    Read a molecule file in various formats (sdf, mol, mol2, pdb).

    Args:
        mol_path: Path to the molecule file

    Returns:
        RDKit molecule object or None if reading fails
    """
    if not os.path.exists(mol_path):
        logger.error(f"Molecule file does not exist: {mol_path}")
        return None

    file_extension = os.path.splitext(mol_path)[1].lower()

    try:
        if file_extension == ".sdf":
            mols = Chem.SDMolSupplier(mol_path, removeHs=True, sanitize=True)
            if len(mols) > 0 and mols[0] is not None:
                return mols[0]

        elif file_extension == ".mol":
            return Chem.MolFromMolFile(mol_path)

        elif file_extension == ".mol2":
            return Chem.MolFromMol2File(mol_path)

        elif file_extension == ".pdb":
            return Chem.MolFromPDBFile(mol_path)

        # If we got here with sanitization and failed, try without sanitizing
        if file_extension == ".sdf":
            mols = Chem.SDMolSupplier(mol_path, removeHs=True, sanitize=False)
            if len(mols) > 0:
                return mols[0]

        logger.error(f"Unsupported molecule format or failed to read: {mol_path}")
        return None
    except Exception as e:
        logger.error(f"Error reading molecule file {mol_path}: {str(e)}")
        return None


def get_molecule_name(mol_path: str) -> str:
    """
    Extract molecule name from the file path.

    Args:
        mol_path: Path to the molecule file

    Returns:
        Molecule name derived from the file path
    """
    base_name = os.path.basename(mol_path)
    molecule_name = os.path.splitext(base_name)[0]
    return molecule_name


def validate_inputs(
    ligand_list: List[str], protein_list: List[str], ref_lig_list: List[str]
) -> bool:
    """
    Validate input files existence and compatibility.

    Args:
        ligand_list: List of ligand file paths
        protein_list: List of protein file paths
        ref_lig_list: List of reference ligand file paths

    Returns:
        True if all inputs are valid, otherwise raises ApoDockError
    """
    if not (len(ligand_list) == len(protein_list) == len(ref_lig_list)):
        raise ApoDockError(
            "Mismatched input lists: ligands, proteins, and reference ligands must have the same count"
        )

    for i, (ligand, protein, ref_lig) in enumerate(
        zip(ligand_list, protein_list, ref_lig_list)
    ):
        if not os.path.exists(ligand):
            raise ApoDockError(f"Ligand file does not exist: {ligand}")
        if not os.path.exists(protein):
            raise ApoDockError(f"Protein file does not exist: {protein}")
        if ref_lig and not os.path.exists(ref_lig):
            raise ApoDockError(f"Reference ligand file does not exist: {ref_lig}")

    return True


def validate_input_files(
    ligand_file: str, protein_file: str, ref_lig_file: Optional[str] = None
) -> bool:
    """
    Validate individual input files existence.

    Args:
        ligand_file: Path to the ligand file
        protein_file: Path to the protein file
        ref_lig_file: Path to the reference ligand file, or None

    Returns:
        True if all inputs are valid, otherwise raises ApoDockError
    """
    if not os.path.exists(ligand_file):
        raise ApoDockError(f"Ligand file does not exist: {ligand_file}")

    if not os.path.exists(protein_file):
        raise ApoDockError(f"Protein file does not exist: {protein_file}")

    if ref_lig_file and not os.path.exists(ref_lig_file):
        raise ApoDockError(f"Reference ligand file does not exist: {ref_lig_file}")

    logger.info("All input files validated successfully")
    return True
