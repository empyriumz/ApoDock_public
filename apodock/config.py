from dataclasses import dataclass, field
from typing import List, Optional
import yaml
from apodock.utils import logger  # Import logger for warning messages


@dataclass
class ModelConfig:
    """Base configuration for ML models."""

    checkpoint_path: str
    device: str = "cuda"


@dataclass
class PackConfig(ModelConfig):
    """Configuration for the Pack model."""

    checkpoint_path: str = "./checkpoints/ApoPack_time_split_0.pt"
    packs_per_design: int = 40
    packing_batch_size: int = 16
    temperature: float = 2.0
    num_clusters: int = 6
    ligandmpnn_path: str = "./checkpoints/proteinmpnn_v_48_020.pt"


@dataclass
class AposcoreConfig(ModelConfig):
    """Configuration for the Aposcore model."""

    checkpoint_path: str = "./checkpoints/ApoScore_time_split_0.pt"
    hidden_dim: int = 256
    num_heads: int = 4
    dropout: float = 0.1
    cross_attention: bool = True
    attention_activation: str = "softmax"
    num_layers: int = 6
    interaction_type: str = "product"
    n_gaussians: int = 10


@dataclass
class DockingEngineConfig:
    """Configuration for GNINA docking program."""

    num_modes: int = 40
    exhaustiveness: int = 32
    autobox_add: float = 6.0
    box_center: Optional[List[float]] = None
    box_size: Optional[List[float]] = None
    gnina_path: str = "/host/gnina"  # Path to GNINA executable


@dataclass
class InputConfig:
    """Configuration for input files."""

    protein_file: Optional[str] = None
    ligand_file: Optional[str] = None
    ref_lig_file: Optional[str] = None


@dataclass
class PipelineConfig:
    """Main configuration for the docking pipeline."""

    input_config: InputConfig = field(default_factory=InputConfig)
    pack_config: PackConfig = field(default_factory=PackConfig)
    aposcore_config: AposcoreConfig = field(default_factory=AposcoreConfig)
    docking_config: DockingEngineConfig = field(default_factory=DockingEngineConfig)
    output_dir: str = "./docking_results"
    top_k: int = 40
    pocket_distance: float = 10.0  # Distance in Angstroms for pocket extraction
    random_seed: int = 42  # Random seed for reproducibility
    # New options moved from command line arguments
    screening_mode: bool = (
        False  # Run in pocket screening mode (outputs scores only, no pose files)
    )
    output_scores_file: str = (
        "pocket_scores.csv"  # File to save pocket scores to (in screening mode)
    )
    save_poses: bool = False  # Save pose files even in screening mode
    rank_by: str = "aposcore"  # Which score to use for ranking pocket designs
    # Options for backbone-only inputs
    skip_pocket_extraction: bool = (
        False  # Skip pocket extraction when already providing a pocket
    )


def load_config_from_yaml(yaml_file: str) -> PipelineConfig:
    """
    Load configuration from a YAML file.

    Args:
        yaml_file: Path to the YAML configuration file

    Returns:
        Complete pipeline configuration
    """
    with open(yaml_file, "r") as f:
        config_dict = yaml.safe_load(f)

    # Create default configuration objects
    input_config = InputConfig()
    pack_config = PackConfig()
    aposcore_config = AposcoreConfig()
    docking_config = DockingEngineConfig()

    # Update with values from YAML
    if "input" in config_dict:
        input_dict = config_dict["input"]
        if "protein_file" in input_dict:
            input_config.protein_file = input_dict["protein_file"]
        if "ligand_file" in input_dict:
            input_config.ligand_file = input_dict["ligand_file"]
        if "ref_lig_file" in input_dict:
            input_config.ref_lig_file = input_dict["ref_lig_file"]

    if "pack" in config_dict:
        pack_dict = config_dict["pack"]
        if "checkpoint_path" in pack_dict:
            pack_config.checkpoint_path = pack_dict["checkpoint_path"]
        if "device" in pack_dict:
            pack_config.device = pack_dict["device"]
        if "packs_per_design" in pack_dict:
            pack_config.packs_per_design = pack_dict["packs_per_design"]
        if "packing_batch_size" in pack_dict:
            pack_config.packing_batch_size = pack_dict["packing_batch_size"]
        if "temperature" in pack_dict:
            pack_config.temperature = pack_dict["temperature"]
        if "num_clusters" in pack_dict:
            pack_config.num_clusters = pack_dict["num_clusters"]
        if "ligandmpnn_path" in pack_dict:
            pack_config.ligandmpnn_path = pack_dict["ligandmpnn_path"]

    if "aposcore" in config_dict:
        aposcore_dict = config_dict["aposcore"]
        if "checkpoint_path" in aposcore_dict:
            aposcore_config.checkpoint_path = aposcore_dict["checkpoint_path"]
        if "device" in aposcore_dict:
            aposcore_config.device = aposcore_dict["device"]
        if "hidden_dim" in aposcore_dict:
            aposcore_config.hidden_dim = aposcore_dict["hidden_dim"]
        if "num_heads" in aposcore_dict:
            aposcore_config.num_heads = aposcore_dict["num_heads"]
        if "dropout" in aposcore_dict:
            aposcore_config.dropout = aposcore_dict["dropout"]
        if "cross_attention" in aposcore_dict:
            aposcore_config.cross_attention = aposcore_dict["cross_attention"]
        if "attention_activation" in aposcore_dict:
            aposcore_config.attention_activation = aposcore_dict["attention_activation"]
        if "num_layers" in aposcore_dict:
            aposcore_config.num_layers = aposcore_dict["num_layers"]
        if "interaction_type" in aposcore_dict:
            aposcore_config.interaction_type = aposcore_dict["interaction_type"]
        if "n_gaussians" in aposcore_dict:
            aposcore_config.n_gaussians = aposcore_dict["n_gaussians"]

    if "docking" in config_dict:
        docking_dict = config_dict["docking"]
        if "num_modes" in docking_dict:
            docking_config.num_modes = docking_dict["num_modes"]
        if "exhaustiveness" in docking_dict:
            docking_config.exhaustiveness = docking_dict["exhaustiveness"]
        if "autobox_add" in docking_dict:
            docking_config.autobox_add = docking_dict["autobox_add"]
        if "box_center" in docking_dict:
            docking_config.box_center = docking_dict["box_center"]
        if "box_size" in docking_dict:
            docking_config.box_size = docking_dict["box_size"]
        if "gnina_path" in docking_dict:
            docking_config.gnina_path = docking_dict["gnina_path"]

    # Create the pipeline configuration
    pipeline_config = PipelineConfig(
        input_config=input_config,
        pack_config=pack_config,
        aposcore_config=aposcore_config,
        docking_config=docking_config,
    )

    # Set pipeline-level parameters
    if "output_dir" in config_dict:
        pipeline_config.output_dir = config_dict["output_dir"]
    if "use_packing" in config_dict:
        # Deprecated parameter, warn user
        logger.warning(
            "The 'use_packing' parameter is deprecated and will be ignored. "
            "Packing is now automatically performed for backbone-only structures."
        )
    if "top_k" in config_dict:
        pipeline_config.top_k = config_dict["top_k"]
    if "pocket_distance" in config_dict:
        pipeline_config.pocket_distance = config_dict["pocket_distance"]
    if "random_seed" in config_dict:
        pipeline_config.random_seed = config_dict["random_seed"]

    # Set new options moved from command line arguments
    if "screening_mode" in config_dict:
        pipeline_config.screening_mode = config_dict["screening_mode"]
    if "output_scores_file" in config_dict:
        pipeline_config.output_scores_file = config_dict["output_scores_file"]
    if "save_poses" in config_dict:
        pipeline_config.save_poses = config_dict["save_poses"]
    if "rank_by" in config_dict:
        # Validate rank_by value
        valid_rank_options = [
            "aposcore",
            "gnina_affinity",
            "gnina_cnn_score",
            "gnina_cnn_affinity",
        ]
        if config_dict["rank_by"] in valid_rank_options:
            pipeline_config.rank_by = config_dict["rank_by"]
        else:
            logger.warning(
                f"Invalid rank_by value: {config_dict['rank_by']}. Using default: aposcore"
            )

    # Set new options for backbone-only inputs
    if "skip_pocket_extraction" in config_dict:
        pipeline_config.skip_pocket_extraction = config_dict["skip_pocket_extraction"]
    if "input_is_backbone_only" in config_dict:
        # Inform user this parameter is now automatically handled
        logger.warning(
            "The 'input_is_backbone_only' parameter has been removed. "
            "The pipeline now automatically detects backbone-only structures."
        )

    return pipeline_config
