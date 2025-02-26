import os
import logging
from typing import Optional, Any, Union, List, Dict
import pandas as pd
from rdkit import Chem
import torch
import random
import numpy as np
import re

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("apodock")


def set_random_seed(seed: int, log: bool = True) -> None:
    """
    Set random seed for all random number generators to ensure reproducibility.

    This function sets the seed for:
    - Python's built-in random module
    - NumPy's random number generator
    - PyTorch's random number generator (both CPU and CUDA)

    Args:
        seed: Integer seed for random number generators
        log: Whether to log the seed setting (default: True)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set environment variable for external programs like GNINA
    os.environ["PYTHONHASHSEED"] = str(seed)

    if log:
        logger.info(f"Random seed set to {seed} for reproducibility")


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


def parse_gnina_scores(output_text: str) -> List[Dict[str, float]]:
    """
    Parse GNINA docking scores from the output text.

    Args:
        output_text: GNINA output text

    Returns:
        List of dictionaries containing scores for each pose
    """
    scores = []
    lines = output_text.split("\n")

    # Find the score table
    header = "mode |  affinity  |  intramol  |    CNN     |   CNN"

    for i, line in enumerate(lines):
        if line.strip() == header:
            # Skip the header, subheader, and separator lines
            score_lines = lines[i + 3 :]  # Start after separator line
            for line in score_lines:
                if not line.strip():  # Stop at empty line
                    break
                try:
                    # Split on whitespace and extract values
                    parts = line.strip().split()
                    scores.append(
                        {
                            "affinity": float(parts[1]),  # Vina score
                            "intramol": float(parts[2]),  # Intramolecular score
                            "cnn_score": float(parts[3]),  # CNN pose score
                            "cnn_affinity": float(parts[4]),  # CNN affinity
                        }
                    )
                except (IndexError, ValueError) as e:
                    if not line.strip().startswith("WARNING"):
                        logger.warning(f"Could not parse score line: {line.strip()}")
                    continue
            break

    if not scores:
        logger.warning("No valid scores found in GNINA output")

    return scores


def extract_gnina_scores_from_file(sdf_file: str) -> Optional[List[Dict[str, float]]]:
    """
    Extract GNINA scores from an SDF file by looking for the original GNINA output
    in the same directory.

    Args:
        sdf_file: Path to the SDF file containing docked poses

    Returns:
        List of dictionaries containing scores for each pose, or None if not found
    """
    # Get the directory and base filename
    directory = os.path.dirname(sdf_file)
    base_name = os.path.basename(sdf_file).split(".")[0]

    # Look for a log file with the same base name
    log_file = os.path.join(directory, f"{base_name}.log")

    if not os.path.exists(log_file):
        # Try to find any log file in the directory
        log_files = [f for f in os.listdir(directory) if f.endswith(".log")]
        if log_files:
            log_file = os.path.join(directory, log_files[0])
        else:
            logger.warning(f"No GNINA log file found for {sdf_file}")
            return None

    try:
        with open(log_file, "r") as f:
            log_content = f.read()

        # Parse the scores from the log content
        scores = parse_gnina_scores(log_content)
        return scores
    except Exception as e:
        logger.warning(f"Error extracting GNINA scores from log file: {str(e)}")
        return None


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
