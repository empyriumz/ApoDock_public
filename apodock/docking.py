import os
import sys
import argparse
import subprocess
import numpy as np
from rdkit import Chem
import pandas as pd
from apodock.Pack_sc.inference import sc_pack
from apodock.Pack_sc.Packnn import Pack
from apodock.Aposcore.inference_dataset import get_mdn_score, read_sdf_file
from apodock.Aposcore.Aposcore import Aposcore

def rank_docking_results(
    docked_sdfs, mdn_scores, top_k=10, packing=True, docking_program="smina"
):
    if docking_program == "smina.static":
        docking_program = "smina"
    else:
        docking_program = docking_program

    mol_list = []
    mol_name_list = []
    for sdf in docked_sdfs:
        mols, mol_names = read_sdf_file(sdf, save_mols=True)
        if mols is None:
            continue
        mol_list.extend(mols)
        mol_name_list.extend(mol_names)
    # print(len(mol_list))
    if len(mol_list) == 0:
        print(
            "No molecules were read from the SDF files or pass the rdkit sanitization"
        )
        return None

    assert len(mol_list) == len(mol_name_list), "Molecule and name counts do not match."
    assert len(mol_list) == len(mdn_scores), "Score and molecule counts do not match."

    # Sort the docked poses based on the mdn scores, and synchronize names with molecules
    sorted_pairs = sorted(
        zip(mdn_scores, mol_list, mol_name_list), key=lambda pair: pair[0], reverse=True
    )
    sorted_mol_list = [mol for _, mol, _ in sorted_pairs]
    sorted_mol_name_list = [name for _, _, name in sorted_pairs]

    # Write the sorted mol_list to an sdf file (top k poses)
    top_k_mol_list = sorted_mol_list[:top_k]
    top_k_mol_name_list = sorted_mol_name_list[:top_k]

    if packing:
        top_k_sdf_filename = f"Packed_top_{top_k}_{docking_program}_rescore_poses.sdf"
    else:
        top_k_sdf_filename = f"Top_{top_k}_{docking_program}_rescore_poses.sdf"
    top_k_sdf = os.path.join(os.path.dirname(docked_sdfs[0]), top_k_sdf_filename)

    w = Chem.SDWriter(top_k_sdf)
    for i, mol in enumerate(top_k_mol_list):
        mol.SetProp("_Name", top_k_mol_name_list[i])
        w.write(mol)
    w.close()

    return top_k_sdf


def get_data_from_csv(csv_file):
    """
    This function reads the CSV file containing the ligand and protein paths.
    """
    data = pd.read_csv(csv_file)
    ligand_list = data["ligand"].tolist()
    protein_list = data["protein"].tolist()
    ref_lig_list = data["ref_lig"].tolist()

    return ligand_list, protein_list, ref_lig_list


def vina_dock(
    ligand,
    protein,
    docking_program,  # or "gnina"
    ref_lig,
    box_center,
    box_size,
    exhaustiveness,
    num_modes,
    autobox_add,
    packing,
    out_dir,
):
    """
    This function docks a ligand to a protein using gnina.
    """
    ligand_name = os.path.basename(ligand).split(".")[0]
    protein_name = os.path.basename(protein).split(".")[0]
    ligand_id = ligand_name.split("_ligand")[0]
    protein_id = protein_name.split("_protein")[0]
    if not os.path.exists(out_dir):
        os.makedirs(os.path.join(out_dir, exist_ok=True))

    if packing:
        # output_dir = os.path.join(data_path, f"{protein_name.split('.')[0]}_dock_{ligand_name.split('_')[0]}.sdf")
        if docking_program == "smina.static":
            output_dir = os.path.join(
                out_dir, f"{protein_id.split('.')[0]}_smina_dock_{ligand_id}.sdf"
            )
        elif docking_program == "gnina":
            output_dir = os.path.join(
                out_dir, f"{protein_id.split('.')[0]}_gnina_dock_{ligand_id}.sdf"
            )
    else:
        # protein_path = protein_name
        if docking_program == "smina.static":
            output_dir = os.path.join(out_dir, f"smina_dock_{ligand_id}.sdf")
        elif docking_program == "gnina":
            output_dir = os.path.join(out_dir, f"gnina_dock_{ligand_id}.sdf")
    GNINA_PATH = "/host/gnina"
    if ref_lig:
        # ref_lig_path = os.path.join(data_path, ref_lig)

        gnina_cmd = f"{GNINA_PATH} --receptor {protein} --ligand {ligand} --autobox_ligand {ref_lig} \
            --autobox_add {autobox_add} --num_modes {num_modes} --exhaustiveness {exhaustiveness} --out {output_dir}"
    else:
        if not np.all(box_center) or not np.all(box_size):
            sys.exit("The box center and size must be provided")

        gnina_cmd = f"{GNINA_PATH} -r {protein} -l {ligand} --center_x {box_center[0]} \
            --center_y {box_center[1]} --center_z {box_center[2]} --size_x {box_size[0]}  \
                --size_y {box_size[1]} --size_z {box_size[2]} --num_modes {num_modes} \
                    -o {output_dir} --exhaustiveness {exhaustiveness}"

    subprocess.run(gnina_cmd, shell=True)

    return output_dir


def flex_docking(
    ligand_list,
    pocket_list,
    protein_list,
    ref_lig_list,
    model_sc,
    ckpt_sc,
    packs_per_design,
    docking_program,
    packing,
    packing_batch_size,
    temperature,
    num_clusters,
    ligandmpnn_path,
    model_mdn,
    ckpt_mdn,
    auto_box_add,
    box_center,
    box_size,
    num_modes,
    exhaustiveness,
    device,
    top_k,
    out_dir,
):
    """
    This function docks a ligand to a protein using the flexible docking approach.
    """
    ids_list = [os.path.basename(i).split("_ligand")[0] for i in ligand_list]

    if packing:
        cluster_packs_list = sc_pack(
            ligand_list,
            pocket_list,
            protein_list,
            model_sc,
            ckpt_sc,
            device,
            packing_batch_size,
            packs_per_design,
            out_dir,
            temperature,
            ligandmpnn_path,
            apo2holo=False,
            num_clusters=num_clusters,
        )
        print("packing complete")

    for ids in ids_list:
        print(f"Docking {ids}....")
        ids_dir = os.path.dirname(pocket_list[ids_list.index(ids)])

        if ref_lig_list is not None:
            ref_lig = ref_lig_list[ids_list.index(ids)]
        else:
            ref_lig = None
        ligand = ligand_list[ids_list.index(ids)]
        if packing:
            packed_pockets = next(
                (i for i in cluster_packs_list if any(ids in j for j in i)), None
            )

            out_sdfs = []
            # remove old docked sdf files
            old_files = [
                os.path.join(ids_dir, i)
                for i in os.listdir(ids_dir)
                if i.endswith(".sdf") and "_pack_" in i
            ]

            for old_file in old_files:
                os.remove(old_file)
            for packed_pdb in packed_pockets:
                out_sdf = vina_dock(
                    ligand,
                    packed_pdb,
                    docking_program,
                    ref_lig,
                    box_center,
                    box_size,
                    exhaustiveness,
                    num_modes,
                    auto_box_add,
                    packing,
                    out_dir=ids_dir,
                )
                out_sdfs.append(out_sdf)
            print("Starting to score the docked poses....")
            scores = get_mdn_score(
                out_sdfs, packed_pockets, model_mdn, ckpt_mdn, device, dis_threshold=5.0
            )
            rank_docking_results(
                out_sdfs,
                scores,
                top_k=top_k,
                packing=packing,
                docking_program=docking_program,
            )

        else:
            print("only docking and rescoring the docked poses....")

            packed_pdb = os.path.join(ids_dir, "Pocket_10A.pdb")
            out_sdf = vina_dock(
                ligand,
                packed_pdb,
                docking_program,
                ref_lig,
                box_center,
                box_size,
                exhaustiveness,
                num_modes,
                auto_box_add,
                packing,
                out_dir=ids_dir,
            )
            print("Starting to score the docked poses....")

            scores = get_mdn_score(
                [out_sdf], [packed_pdb], model_mdn, ckpt_mdn, device, dis_threshold=5.0
            )

            print("Ranking the docked poses....")
            rank_docking_results(
                [out_sdf],
                scores,
                top_k=top_k,
                packing=packing,
                docking_program=docking_program,
            )

    print("docking complete")


def get_pocket(ref_ligand, protein, out_dir, distance=10):
    """
    This function gets the pocket of the protein using pymol
    """
    import pymol

    pymol.cmd.load(protein, "protein")
    pymol.cmd.remove("resn HOH")
    pymol.cmd.remove("not polymer.protein")
    pymol.cmd.load(ref_ligand, "ligand")
    pymol.cmd.remove("hydrogens")
    pymol.cmd.select("Pocket", f"byres ligand around {distance}")

    pocket_path = os.path.join(out_dir, f"Pocket_{distance}A.pdb")
    pymol.cmd.save(pocket_path, "Pocket")
    pymol.cmd.delete("all")

    return pocket_path


def get_pocket_list(protein_list, ref_lig_list, out_dir, distance=10):
    out_dirs = []
    for protein, ref_lig in zip(protein_list, ref_lig_list):
        id = os.path.basename(protein).split("_protein")[0]
        out_put_dir = os.path.join(out_dir, id)
        os.makedirs(out_put_dir, exist_ok=True)
        out_put_pocket = get_pocket(ref_lig, protein, out_put_dir, distance=distance)
        out_dirs.append(out_put_pocket)
    # print(out_dirs)
    return out_dirs


def parse_args():
    parser = argparse.ArgumentParser(description="Docking with ApoDock")
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to the CSV file containing the ligand and protein paths",
    )
    parser.add_argument(
        "--protein", type=str, help="Path to the directory containing the proteins"
    )
    parser.add_argument(
        "--ligand", type=str, help="Path to the directory containing the ligands"
    )
    parser.add_argument(
        "--ref_lig", type=str, default=None, help="Path to the reference ligand"
    )
    parser.add_argument(
        "--ckpt_sc",
        type=str,
        default="./checkpoints/ApoPack_time_split_0.pt",
        help="Path to the ckpt of the SC model",
    )
    parser.add_argument(
        "--ckpt_mdn",
        type=str,
        default="./checkpoints/ApoScore_time_split_0.pt",
        help="Path to the ckpt of the MDN model",
    )
    parser.add_argument(
        "--packs_per_design", type=int, default=40, help="Number of packs per design"
    )
    parser.add_argument(
        "--docking_program", type=str, default="gnina", help="Docking program to use"
    )
    parser.add_argument(
        "--packing", action="store_true", help="Whether to pack the protein"
    )
    parser.add_argument(
        "--packing_batch", type=int, default=16, help="Batch size for packing"
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=6,
        help="Number of clusters to use for packing",
    )
    parser.add_argument(
        "--ligandmpnn_path",
        type=str,
        default="./checkpoints/proteinmpnn_v_48_020.pt",
        help="Path to the ligand MPNN model",
    )
    parser.add_argument(
        "--num_modes", type=int, default=40, help="Number of modes to use for docking"
    )
    parser.add_argument(
        "--exhaustiveness",
        type=int,
        default=32,
        help="Exhaustiveness to use for docking",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for docking"
    )
    parser.add_argument(
        "--top_k", type=int, default=40, help="Number of top poses to keep"
    )
    parser.add_argument(
        "--autobox_add", type=float, default=6.0, help="Autobox add value"
    )
    parser.add_argument(
        "--temperature", type=float, default=2, help="Temperature value for packing"
    )
    parser.add_argument(
        "--box_center", type=list, default=None, help="Center of the box"
    )
    parser.add_argument("--box_size", type=list, default=None, help="Size of the box")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./docking_results",
        help="Output directory for docking results",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.csv:
        ligand_list, protein_list, ref_lig_list = get_data_from_csv(args.csv)
    else:
        ligand_list = [args.ligand]
        protein_list = [args.protein]
        ref_lig_list = [args.ref_lig]
    # print(ligand_list)
    # print(len(ligand_list), len(protein_list), len(ref_lig_list))
    assert (
        len(ligand_list) == len(protein_list) == len(ref_lig_list)
    ), "Data counts do not match"
    print("whether packing:", args.packing)
    pocket_list = get_pocket_list(protein_list, ref_lig_list, args.out_dir)
    model_sc = Pack(recycle_strategy="sample")
    model_mdn = Aposcore(
        35,
        hidden_dim=256,
        num_heads=4,
        dropout=0.1,
        crossAttention=True,
        atten_active_fuc="softmax",
        num_layers=6,
        interact_type="product",
    )

    flex_docking(
        ligand_list,
        pocket_list,
        protein_list,
        ref_lig_list,
        model_sc,
        args.ckpt_sc,
        args.packs_per_design,
        args.docking_program,
        args.packing,
        args.packing_batch,
        args.temperature,
        args.num_clusters,
        args.ligandmpnn_path,
        model_mdn,
        args.ckpt_mdn,
        args.autobox_add,
        args.box_center,
        args.box_size,
        args.num_modes,
        args.exhaustiveness,
        args.device,
        args.top_k,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
