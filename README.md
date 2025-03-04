[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/ld139/ApoDock_public)
[![bioRxiv](https://img.shields.io/badge/bioRxiv2024.11.22.624942-green)](https://doi.org/10.1101/2024.11.22.624942)


# ApoDock
<div align=center>
<img src='./toc.svg' width='600',height="300px">
</div> 


ApoDock is a modular docking paradigm that combines machine learning-driven conditional side-chain packing based on protein backbone and ligand information with traditional sampling methods to ensure physically realistic poses.

To run ApoDock, clone this GitHub repository and install Python.

## Requirements

-  Operating System: Linux (Recommended)

## Install Dependencies
```
conda env create -f environment.yaml
conda activate apodock

```

## Quick start

First, you need to download the binary sampling software [Gnina](https://github.com/gnina/gnina/releases/download/v1.1/gnina) or [Smina](https://sourceforge.net/projects/smina/) to `docking_program` dir, then give execution permission:
```
chmod +x Gnina
chmod +x Smina.static
```
Then set the enviroment path:
```
export PATH="$PATH:/your/path/to/ApoDock_public/docking_program"
```

## Configuration-based Approach

ApoDock now uses a YAML configuration file for all settings. This provides a centralized, flexible way to configure the docking pipeline.

### Basic Configuration

Create a configuration file (e.g., `config.yaml`):

```yaml
# General pipeline settings
output_dir: ./docking_results
top_k: 40
pocket_distance: 10.0
random_seed: 42

# Input file settings
input:
  protein_file: ./demo/1a0q/1a0q_protein.pdb
  ligand_file: ./demo/1a0q/1a0q_ligand.sdf
  ref_lig_file: ./demo/1a0q/1a0q_ligand.sdf

# Docking engine settings
docking:
  program: gnina
  gnina_path: /path/to/gnina
```

### Running ApoDock

Run ApoDock with your configuration file:

```bash
python -m apodock.docking --config config.yaml
```

To view the loaded configuration without running the pipeline:

```bash
python -m apodock.docking --config config.yaml --show_config
```

### Pipeline Workflow

ApoDock follows this streamlined, intelligent workflow:

1. **Protein Structure Validation**:
   - The pipeline automatically detects whether input protein structures are backbone-only (containing only N, CA, C, O atoms) or full-atom models.
   - This detection happens without any manual configuration flags.

2. **Pocket Extraction**:
   - The pipeline extracts the pocket region from the input protein structure based on the reference ligand position and the configured pocket distance.
   - For pre-extracted pockets, this step can be skipped by setting `skip_pocket_extraction: true` in the configuration.

3. **Side-Chain Packing** (automatically applied for backbone-only structures):
   - When a structure is detected as backbone-only, multiple packed protein variants are automatically generated with different side-chain conformations.
   - These variants are named with the pattern `{protein_id}_pack_{number}` (e.g., "1a0q_pack_24").
   - For full-atom models, packing is skipped, and the original structure is used directly.

4. **Docking**: 
   - For backbone-only structures: The ligand is docked to all generated packed variants.
   - For full-atom structures: The ligand is docked to the original structure with existing side chains.

5. **Scoring and Ranking**:
   - Each docked pose is evaluated using multiple scoring metrics:
     - ApoScore: Machine learning-based scoring function
     - GNINA scores: Affinity, CNN score, and CNN affinity
   - Scores are organized by structure ID, with each structure having multiple pose scores
   - The best structure is selected based on the highest ApoScore (default) or other specified metrics
   - Only the best structure and its corresponding scores are saved in the final output

This approach simplifies the workflow while maintaining all functionality - ApoDock intelligently decides when to apply side-chain packing based on the input structure type.

### Using ApoDock with Backbone-Only Pockets

ApoDock works seamlessly with backbone-only pockets (structures containing only N, CA, C, O atoms):

1. **Input Requirements**:
   - Provide the backbone-only pocket as `protein_file`.
   - Provide the ligand as `ligand_file`.
   - A reference ligand (`ref_lig_file`) is needed to define the binding site location.

2. **Pipeline Behavior with Backbone-Only Inputs**:
   - The pipeline automatically detects that your input is a backbone-only structure.
   - Side-chain packing is automatically applied to generate realistic protein models.
   - Results will show entries with the `_pack_#` suffix, representing different packed variants.

3. **Example Configuration for Backbone-Only Input**:
   ```yaml
   # General settings
   output_dir: ./backbone_docking_results
   
   # Input files - backbone-only protein
   input:
     protein_file: ./demo/backbone_only/1a0q_backbone.pdb
     ligand_file: ./demo/backbone_only/1a0q_ligand.sdf
     ref_lig_file: ./demo/backbone_only/1a0q_ligand.sdf
   
   # Skip pocket extraction if already providing a pocket
   skip_pocket_extraction: true
   ```

4. **Important Note**:
   For backbone-only inputs, it's often beneficial to set `skip_pocket_extraction: true` if you're already providing a pre-extracted pocket.

### Protein Structure Requirements

ApoDock performs automatic validation of protein structure completeness:

- Both complete proteins and backbone-only structures are accepted without any special configuration.
- For backbone-only structures, the packing algorithm will automatically generate side chains.
- No manual flags or special settings are required - the pipeline intelligently adapts to the input structure type.

### Pocket Screening Mode

For pocket design screening, create a configuration with screening options:

```yaml
# General settings
output_dir: ./screening_results

# Screening options
screening_mode: true
save_poses: false
rank_by: aposcore  # Options: aposcore, gnina_affinity, gnina_cnn_score, gnina_cnn_affinity

# Input files
input:
  protein_file: ./demo/1a0q/1a0q_protein.pdb
  ligand_file: ./demo/1a0q/1a0q_ligand.sdf
  ref_lig_file: ./demo/1a0q/1a0q_ligand.sdf
```

### Multiple Configuration Files

Create different configuration files for different use cases:

- `standard_docking.yaml` - For standard docking with pose generation
- `screening_aposcore.yaml` - For pocket screening with ApoScore ranking
- `screening_gnina.yaml` - For pocket screening with GNINA score ranking

## Output

Results will be saved to the directory specified in the configuration file (`output_dir`). 

When running in screening mode:
- A CSV file with ranking information will be generated
- For each protein, only the best structure is saved, based on the ranking metric
- The output includes:
  - Best structure files: `{protein_id}_best_{rank_by}_protein.pdb` and `{protein_id}_best_{rank_by}_ligand.sdf`
  - Comprehensive scores dictionary containing:
    ```
    {protein_id}: {
        "aposcore": float,
        "gnina_affinity": float,
        "gnina_cnn_score": float,
        "gnina_cnn_affinity": float
    }
    ```
- The ranking output shows the best structure for each protein with all its scores:
  ```
  Protein: 1a0q_pack_24
    ApoScore: 47.59 (higher is better)
    GNINA Affinity: -7.04 (lower is better)
    GNINA CNN Score: 0.89 (higher is better)
    GNINA CNN Affinity: -6.92 (lower is better)
  ```
- Only the top-performing structure (based on the specified ranking metric) is saved
- Higher rankings for packed variants indicate successful side-chain optimization

## Advanced Configuration

See the sample configuration file in `apodock/configs/config.yaml` for all available options and their descriptions.

## Acknowledgements
This work draws upon code from [ProteinMPNN](https://github.com/dauparas/ProteinMPNN), [OpenFold](https://github.com/aqlaboratory/openfold), [RTMscore](https://github.com/sc8668/RTMScore), and [PIPPack](https://github.com/Kuhlman-Lab/PIPPack), and we would like to thank them for their excellent contributions. these studies are important and interesting.

