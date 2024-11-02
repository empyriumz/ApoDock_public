# ApoDock




ApoDock is a modular docking paradigm that combines machine learning-driven conditional side-chain packing based on protein backbone and ligand information with traditional sampling methods to ensure physically realistic poses.

To run ApoDock, clone this GitHub repository and install Python.

## Requirements

-  Operating System: Linux (Recommended)

## Install Dependencies
```
conda env create -f Enviroment.yaml
conda activate apodock

```
Then install gvp-gnn:
```
git clone https://github.com/drorlab/gvp-pytorch.git
cd gvp-pytorch
pip install -e .
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
python docking_esm_protein.py --csv docking_list.csv --packing
```
-----------------------------------------------------------------------------------------------------
Output example will in defalut dir `docking_results`, out you can use `--out_dir` option to determine the output position.


-----------------------------------------------------------------------------------------------------

