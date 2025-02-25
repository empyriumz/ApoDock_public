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

## Inference

A demo:
```
python docking.py --protein ./demo/1a0q/1a0q_protein.pdb --ligand ./demo/1a0q/1a0q_ligand.sdf --ref_lig ./demo/1a0q/1a0q_ligand.sdf --packing
```
Use `.CSV` file for docking:
```
python docking.py --csv docking_list.csv --packing
```
-----------------------------------------------------------------------------------------------------
Output example will in defalut dir `docking_results`,  you can use `--out_dir` option to determine the output position.


-----------------------------------------------------------------------------------------------------

## Acknowledgements
This work draws upon code from [ProteinMPNN](https://github.com/dauparas/ProteinMPNN), [OpenFold](https://github.com/aqlaboratory/openfold), [RTMscore](https://github.com/sc8668/RTMScore), and [PIPPack](https://github.com/Kuhlman-Lab/PIPPack), and we would like to thank them for their excellent contributions. these studies are important and interesting.

