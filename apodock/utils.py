import os
import logging
from typing import Optional, Any, Union
import pandas as pd
from rdkit import Chem
import torch

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("apodock")


class ApoDockError(Exception):
    """Custom exception for ApoDock errors."""

    pass


def ensure_dir(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory: Directory path to check/create
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Created directory: {directory}")


def get_data_from_csv(csv_file: str) -> tuple:
    """
    Read CSV file containing paths for ligands, proteins, and reference ligands.

    Args:
        csv_file: Path to the CSV file

    Returns:
        Tuple of (ligand_list, protein_list, ref_lig_list)

    Raises:
        ApoDockError: If counts mismatch or if error occurs
    """
    try:
        df = pd.read_csv(csv_file)
        if "ligand" not in df.columns or "protein" not in df.columns:
            raise ApoDockError(
                f"CSV file must contain 'ligand' and 'protein' columns: {csv_file}"
            )

        ligand_list = df["ligand"].tolist()
        protein_list = df["protein"].tolist()

        # Reference ligand is optional
        if "ref_lig" in df.columns:
            ref_lig_list = df["ref_lig"].tolist()
        else:
            ref_lig_list = [None] * len(ligand_list)

        # Check if counts match
        if len(ligand_list) != len(protein_list) or len(ligand_list) != len(
            ref_lig_list
        ):
            raise ApoDockError(
                f"Mismatched counts in CSV file: ligands ({len(ligand_list)}), "
                f"proteins ({len(protein_list)}), ref_ligs ({len(ref_lig_list)})"
            )

        return ligand_list, protein_list, ref_lig_list

    except Exception as e:
        if isinstance(e, ApoDockError):
            raise
        raise ApoDockError(f"Error reading CSV file {csv_file}: {str(e)}")


def read_molecule(mol_path: str) -> Optional[Chem.Mol]:
    """
    Read a molecule file in various formats (sdf, mol, mol2, pdb).

    Args:
        mol_path: Path to the molecule file

    Returns:
        RDKit molecule object or None if reading fails
    """
    if not os.path.exists(mol_path):
        logger.error(f"Molecule file not found: {mol_path}")
        return None

    file_ext = os.path.splitext(mol_path)[1].lower()

    try:
        if file_ext in [".sdf", ".mol"]:
            return Chem.SDMolSupplier(mol_path, sanitize=True, removeHs=False)[0]
        elif file_ext == ".mol2":
            return Chem.MolFromMol2File(mol_path, sanitize=True, removeHs=False)
        elif file_ext == ".pdb":
            return Chem.MolFromPDBFile(mol_path, sanitize=True, removeHs=False)
        else:
            logger.error(f"Unsupported molecule file format: {file_ext}")
            return None
    except Exception as e:
        logger.error(f"Error reading molecule file {mol_path}: {str(e)}")
        return None


def get_molecule_name(mol_path: str) -> str:
    """
    Extract molecule name from file path.

    Args:
        mol_path: Path to the molecule file

    Returns:
        Molecule name as string
    """
    return os.path.splitext(os.path.basename(mol_path))[0]


def validate_input_files(
    ligand_file: str, protein_file: str, ref_lig_file: Optional[str] = None
) -> None:
    """
    Validate that input files exist and are compatible.

    Args:
        ligand_file: Path to the ligand file
        protein_file: Path to the protein file
        ref_lig_file: Path to the reference ligand file (optional)

    Raises:
        ApoDockError: If any file does not exist or lists are mismatched
    """
    # Check if files exist
    if not os.path.exists(ligand_file):
        raise ApoDockError(f"Ligand file not found: {ligand_file}")

    if not os.path.exists(protein_file):
        raise ApoDockError(f"Protein file not found: {protein_file}")

    if ref_lig_file and not os.path.exists(ref_lig_file):
        raise ApoDockError(f"Reference ligand file not found: {ref_lig_file}")


class ModelManager:
    """
    Handles model loading, device management, and inference setup across the codebase.
    This provides a consistent way to load models and manage devices for inference.
    """

    @staticmethod
    def get_device(device_str: Optional[str] = None) -> torch.device:
        """
        Get the appropriate torch device.

        Args:
            device_str: Device string ('cuda', 'cuda:0', 'cpu', etc.)

        Returns:
            torch.device: The device to use for computation
        """
        if device_str is None:
            device_str = "cuda" if torch.cuda.is_available() else "cpu"

        if device_str.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Using CPU instead.")
            device_str = "cpu"

        return torch.device(device_str)

    @staticmethod
    def load_model(
        model: Any, checkpoint_path: str, device: Union[str, torch.device]
    ) -> Any:
        """
        Load a model from a checkpoint file and prepare it for inference.

        Args:
            model: The model object to load weights into
            checkpoint_path: Path to the model checkpoint
            device: Device to run the model on ('cpu', 'cuda:0', etc.)

        Returns:
            The loaded model placed on the specified device in eval mode

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If model loading fails
        """
        try:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

            logger.info(f"Loading model from {checkpoint_path}")

            # Simple model loading approach
            model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

            # Move to device and set to evaluation mode
            if isinstance(device, str):
                device = ModelManager.get_device(device)

            model = model.to(device)
            model.eval()

            return model

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}") from e

    @staticmethod
    def cleanup_gpu_memory(device: Union[str, torch.device]) -> None:
        """
        Clean up GPU memory after model usage.

        Args:
            device: The device that was used for computation
        """
        if (
            isinstance(device, str)
            and device.startswith("cuda")
            or isinstance(device, torch.device)
            and device.type == "cuda"
        ):
            try:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    logger.debug("GPU memory cleared")
            except Exception as e:
                logger.warning(f"Failed to clear GPU memory: {str(e)}")
