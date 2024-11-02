import os

import pymol
import tqdm

def generate_pocket_posebuster(data_dir,distance=10):

    pdbids = os.listdir(data_dir)

    for pdbid in pdbids:
 
        lig_path = os.path.join(data_dir, pdbid, f'{pdbid}_ligand.sdf')
        protein_path = os.path.join(data_dir, pdbid, f'{pdbid}_protein.pdb')

        if os.path.exists(os.path.join(data_dir, pdbid, f'Pocket_{distance}A.pdb')):
            continue

        pymol.cmd.load(protein_path)
        pymol.cmd.remove('resn HOH')
        # clean up the non-protein residues
        pymol.cmd.remove("not polymer.protein")
        

    
        pymol.cmd.load(lig_path)
        pymol.cmd.remove('hydrogens')
        pymol.cmd.select('Pocket', f'byres {pdbid}_ligand around {distance}')
        pymol.cmd.save(os.path.join(data_dir, pdbid, f'Pocket_{distance}A.pdb'), 'Pocket')
        pymol.cmd.delete('all')



def align_pdb(apo, holo):


        # apo_path = os.path.join(data_dir, pdbid, 'apo.pdb')
        # holo_path = os.path.join(data_dir, pdbid, 'refined.pdb')
        # align the apo with refined.pdb
    pymol.cmd.load(apo, "apo")
    pymol.cmd.load(holo, "refined")
    pymol.cmd.align("apo", "refined")

    pymol.cmd.save(os.path.join(os.path.dirname(holo), 'aligned_apo.pdb'), 'apo')
    pymol.cmd.delete('all')

def get_pocket_apo2holo(data_dir, distance=10):

    pdbids = os.listdir(data_dir)

    for pdbid in tqdm.tqdm(pdbids):
        apo = os.path.join(data_dir, pdbid, 'apo.pdb')
        holo = os.path.join(data_dir, pdbid, 'refined.pdb')
        align_pdb(apo, holo)
        ligs = os.listdir(os.path.join(data_dir, pdbid, 'ligands'))
        for lig in ligs:
            lig_name = lig.split('.')[0]
            lig_path = os.path.join(data_dir, pdbid, 'ligands', lig)
            os.makedirs(os.path.join(data_dir, pdbid, "Complex", lig_name), exist_ok=True)
            # os.system(f"cp {lig_path} {os.path.join(data_dir, pdbid, 'Complex', lig_name, f'{lig_name}_ligand.pdb')}")
            os.system(f" rm {os.path.join(data_dir, pdbid, 'Complex', lig_name, f'{lig_name}_ligand.pdb')}")
            pocket_path = os.path.join(data_dir, pdbid, "Complex", lig_name, f"Pocket_{distance}A.pdb")
            ligand_save_path = os.path.join(data_dir, pdbid, 'Complex', lig_name, f'{lig_name}_ligand.sdf')
            if not os.path.exists(os.path.dirname(pocket_path)):
                os.makedirs(os.path.dirname(pocket_path))
            if not os.path.exists(os.path.dirname(ligand_save_path)):
                os.makedirs(os.path.dirname(ligand_save_path))
            # if not os.path.exists(pocket_path):
            pymol.cmd.load(os.path.join(data_dir, pdbid, 'aligned_apo.pdb'))
            pymol.cmd.remove("not polymer.protein")
            pymol.cmd.load(lig_path)
            pymol.cmd.remove('hydrogens')
            pymol.cmd.select('Pocket', f'byres {lig_name} around {distance}')
            pymol.cmd.save(pocket_path, 'Pocket', format='pdb')
            pymol.cmd.save(ligand_save_path, lig_name, format='sdf')
            pymol.cmd.delete('all')

def get_pocket(ligand,protein,distance=10):
    ligand_name = os.path.basename(ligand).split('.')[0]
    pymol.cmd.load(protein)
    pymol.cmd.remove('resn HOH')
    # clean up the non-protein residues
    pymol.cmd.remove("not polymer.protein")
    pymol.cmd.load(ligand)
    pymol.cmd.remove('hydrogens')
    pymol.cmd.select('Pocket', f'byres {ligand_name} around {distance}')
    pocket_path = os.path.join(os.path.dirname(os.path.dirname(ligand)), f"Pocket_{distance}A.pdb")
    pymol.cmd.save(pocket_path, 'Pocket')
    pymol.cmd.delete('all')
    return pocket_path

def get_pocket_cross_docking(data_dir, distance=10):

    pdbpairs = os.listdir(os.path.join(data_dir, "Complex"))

    for pdbpair in tqdm.tqdm(pdbpairs):
        if os.path.exists(os.path.join(data_dir, "Complex", pdbpair, f"Pocket_{distance}A.pdb")):
            continue
        protein_id = pdbpair.split('_')[0]
        ligand_id = pdbpair.split('_')[1]
        pymol_id = ligand_id + "_LIG_aligned"
        protein_path = os.path.join(data_dir, "Complex", pdbpair, f'{protein_id}_PRO.pdb')
        ligand_path = os.path.join(data_dir, "Complex", pdbpair, f'{ligand_id}_LIG_aligned.sdf')
        pocket_path = os.path.join(data_dir, "Complex", pdbpair, f"Pocket_{distance}A.pdb")

        # if not os.path.exists(pocket_path):
        pymol.cmd.load(protein_path)
        pymol.cmd.remove('resn HOH')
        # clean up the non-protein residues
        pymol.cmd.remove("not polymer.protein")
        pymol.cmd.load(ligand_path)
        pymol.cmd.remove('hydrogens')
        pymol.cmd.select('Pocket', f'byres {pymol_id} around {distance}')
        pymol.cmd.save(pocket_path, 'Pocket')
        pymol.cmd.delete('all')

def align_protein_esmfold(pdb, ref_pdb, save_dir):
    pymol.cmd.load(pdb, "protein")
    pymol.cmd.load(ref_pdb, "ref_pdb")
    pymol.cmd.align("protein", "ref_pdb")
    pymol.cmd.save(save_dir, 'protein')
    pymol.cmd.delete('all')

def get_pocket_esmfold(csv_file, data_dir_ref_pdb, data_dir_ref_ligand, data_dir_esm_protein, data_dir_save, distance=10):
    import pandas as pd
    import shutil
    pdbids = pd.read_csv(csv_file)["pdb"].tolist()
    # print(pdbids)
    for pdbid in tqdm.tqdm(pdbids):
        os.makedirs(os.path.join(data_dir_save, pdbid), exist_ok=True)
        # orgin_pdb = os.path.join(data_dir_esm_protein, f"{pdbid}_esm.pdb")
        # if not os.path.exists(orgin_pdb):
 
        pdb_save_dir = os.path.join(data_dir_save, pdbid, f"{pdbid}_esm_protein_aligned.pdb")

        if not os.path.exists(pdb_save_dir):
            print(f"{pdbid} not exists")
            continue


        lig_dir = os.path.join(data_dir_ref_ligand, f"{pdbid}.sdf")
        pocket_path = os.path.join(data_dir_save, pdbid,  f"Pocket_{distance}A.pdb")
        # if os.path.exists(pocket_path):
        #     continue

        shutil.copy(lig_dir, os.path.join(data_dir_save, pdbid))
        # get pocket by lig in esm protein

        pymol.cmd.load(pdb_save_dir,"protein")
        pymol.cmd.remove('resn HOH')
        # clean up the non-protein residues
        pymol.cmd.remove("not polymer.protein")
        pymol.cmd.load(lig_dir, "ligand")
        pymol.cmd.remove('hydrogens')
        pymol.cmd.select('Pocket', f'byres ligand around {distance}')
        pymol.cmd.save(pocket_path, 'Pocket')
        pymol.cmd.delete('all')

def check_rmsd(csv_file, data_dir_ref_pdb, data_dir_save):
    """ check the rmsd between esm protein and ref protein not to large"""
    import pandas as pd
    import numpy as np
    pdbids = pd.read_csv(csv_file)["pdb"].tolist()
    rmsds = []
    for pdbid in tqdm.tqdm(pdbids):
        ref_pdb = os.path.join(data_dir_ref_pdb,  pdbid, f"{pdbid}_protein.pdb")
        esm_pdb = os.path.join(data_dir_save, pdbid, f"{pdbid}_esm_protein.pdb")

        if not os.path.exists(esm_pdb):
            continue
        pymol.cmd.load(ref_pdb, "ref_pdb")
        pymol.cmd.load(esm_pdb, "esm_pdb")
        rmsd = pymol.cmd.align("ref_pdb", "esm_pdb")
        pymol.cmd.delete('all')
        # print(rmsd)
        if rmsd[0] > 4.0:
            print(f"{pdbid} rmsd is {rmsd[0]}")
            # continue

        
















if __name__ == "__main__":

    # data_dir = "./test/apo2holo_datasets"
    # group = ['group1', 'group2', 'group3']
    # for g in group:
    #     data_path = os.path.join(data_dir, g)
    #     get_pocket_apo2holo(data_path, distance=10)

    # data_dir = "./test/apo2holo_datasets"
    # ligand_path = "1gzkA/ligands"
    # pdbid = "1gzkA"
    # protein_path = "./test/apo2holo_datasets/group3/1gzkA/aligned_apo.pdb"

    # ligands = os.listdir(os.path.join(data_dir, 'group3', ligand_path))
    # print(ligands)
    # for lig in ligands:
    #     lig_name = lig.split('.')[0]
    #     lig_path = os.path.join(data_dir, 'group3', pdbid, "ligands", lig)
    #     get_pocket(lig_path, protein_path, distance=10)

    # data_dir = "./test/wierbowski_cd"
    # targets = os.listdir(data_dir)
    # for target in targets:
    #     get_pocket_cross_docking(os.path.join(data_dir, target), distance=10)
    data_dir = "../PLmodel/supervised/data/pdbbind"
    csv_file = "./Pack_sc/data/timesplit/test_timesplit.csv"
    data_dir_ref_pdb = os.path.join(data_dir, "v2020-other-PL")
    data_dir_ref_ligand = os.path.join(data_dir, "renumber_atom_index_same_as_smiles")

    data_dir_esm_protein = "../esmfold/test_timesplit_output"

            
    data_dir_save = "./test/esmfold_protein"

    get_pocket_esmfold(csv_file, data_dir_ref_pdb, data_dir_ref_ligand, data_dir_esm_protein, data_dir_save)
    # check_rmsd(csv_file, data_dir_ref_pdb, data_dir_save)
