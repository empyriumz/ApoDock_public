from dataclasses import dataclass, field
from typing import List, Optional
import yaml


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
    """Configuration for docking programs."""

    program: str = "gnina"
    num_modes: int = 40
    exhaustiveness: int = 32
    autobox_add: float = 6.0
    box_center: Optional[List[float]] = None
    box_size: Optional[List[float]] = None
    gnina_path: str = "/host/gnina"  # Default path, should be configurable


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
    use_packing: bool = True
    top_k: int = 40
    pocket_distance: float = 10.0  # Distance in Angstroms for pocket extraction


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
        if "program" in docking_dict:
            docking_config.program = docking_dict["program"]
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
        pipeline_config.use_packing = config_dict["use_packing"]
    if "top_k" in config_dict:
        pipeline_config.top_k = config_dict["top_k"]
    if "pocket_distance" in config_dict:
        pipeline_config.pocket_distance = config_dict["pocket_distance"]

    return pipeline_config
