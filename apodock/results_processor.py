import os
from typing import List, Dict, Optional, Tuple
import numpy as np
from rdkit import Chem

from apodock.utils import logger, extract_gnina_scores_from_file
from apodock.Aposcore.inference_dataset import read_sdf_file


class ResultsProcessor:
    """Process and rank docking results."""

    def __init__(self, top_k: int = 10):
        """
        Initialize the results processor.

        Args:
            top_k: Number of top poses to keep in the final output
        """
        self.top_k = top_k

    def rank_results(
        self,
        docked_sdfs: List[str],
        scores: List[float],
        use_packing: bool = True,
        docking_program: str = "smina",
    ) -> Optional[str]:
        """
        Rank docking results based on scores and write top poses to a file.

        Args:
            docked_sdfs: List of paths to docked pose SDF files
            scores: List of scores for each pose
            use_packing: Whether packing was used
            docking_program: Name of the docking program used

        Returns:
            Path to the ranked poses file, or None if processing failed
        """
        if docking_program == "smina.static":
            docking_program = "smina"

        # Extract protein ID from the first SDF file path
        if docked_sdfs:
            first_sdf = docked_sdfs[0]
            # The protein ID is typically part of the file path (dir/protein_id/files)
            protein_id = os.path.basename(os.path.dirname(first_sdf))
            output_dir = os.path.dirname(
                first_sdf
            )  # Use the same directory as input files
        else:
            logger.error("No SDF files provided for ranking")
            return None

        # Read all molecules from SDF files
        mol_list = []
        mol_name_list = []
        for sdf in docked_sdfs:
            mols, mol_names = read_sdf_file(sdf, save_mols=True)
            if mols is None:
                continue
            mol_list.extend(mols)
            mol_name_list.extend(mol_names)

        if len(mol_list) == 0:
            logger.error(
                "No molecules were read from the SDF files or passed the RDKit sanitization"
            )
            return None

        # Check that counts match
        if len(mol_list) != len(mol_name_list):
            logger.error(
                f"Mismatched molecule and name counts: {len(mol_list)} molecules, {len(mol_name_list)} names"
            )
            return None

        # Handle potential score count mismatch
        if len(mol_list) != len(scores):
            logger.warning(
                f"Mismatched molecule and score counts: {len(mol_list)} molecules, {len(scores)} scores"
            )

            # Adjust scores list if needed
            if len(scores) > len(mol_list):
                # If we have more scores than molecules, truncate the scores list
                logger.warning(
                    f"Truncating scores list from {len(scores)} to {len(mol_list)}"
                )
                scores = scores[: len(mol_list)]
            else:
                # If we have fewer scores than molecules, we'll duplicate the last score
                # This is a fallback approach to allow processing to continue
                logger.warning(
                    f"Extending scores list from {len(scores)} to {len(mol_list)} by duplicating last score"
                )
                last_score = scores[-1] if scores else 0.0
                scores.extend([last_score] * (len(mol_list) - len(scores)))

        # Sort molecules by scores
        sorted_pairs = sorted(
            zip(scores, mol_list, mol_name_list), key=lambda pair: pair[0], reverse=True
        )

        sorted_mol_list = [mol for _, mol, _ in sorted_pairs]
        sorted_mol_name_list = [name for _, _, name in sorted_pairs]

        # Take top-k poses
        top_k_mol_list = sorted_mol_list[: self.top_k]
        top_k_mol_name_list = sorted_mol_name_list[: self.top_k]

        # Write to output file
        if use_packing:
            top_k_sdf_filename = f"{protein_id}_Packed_top_{self.top_k}_{docking_program}_rescore_poses.sdf"
        else:
            top_k_sdf_filename = (
                f"{protein_id}_Top_{self.top_k}_{docking_program}_rescore_poses.sdf"
            )

        top_k_sdf = os.path.join(output_dir, top_k_sdf_filename)

        with Chem.SDWriter(top_k_sdf) as w:
            for i, mol in enumerate(top_k_mol_list):
                mol.SetProp("_Name", top_k_mol_name_list[i])
                w.write(mol)

        logger.info(f"Wrote top {self.top_k} poses to {top_k_sdf}")

        # Clean up log files if they exist
        for sdf in docked_sdfs:
            log_file = f"{os.path.splitext(sdf)[0]}.log"
            if os.path.exists(log_file):
                try:
                    os.remove(log_file)
                    logger.debug(f"Removed log file: {log_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove log file {log_file}: {str(e)}")

        return top_k_sdf

    def process_screening_results(
        self,
        scores: Dict[str, List[float]],
        docked_ligands: List[str],
        pocket_files: List[str],
        protein_ids: List[str],
        output_dir: str,
        rank_by: str = "aposcore",
        save_best_structures: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Process scores and save only the best structure and scores.

        Args:
            scores: Dictionary mapping structure IDs to lists of pose scores
            docked_ligands: List of paths to docked ligand files
            pocket_files: List of paths to protein pocket files used for docking
            protein_ids: List of protein IDs
            output_dir: Output directory for results
            rank_by: Which score to use for ranking (default: "aposcore")
                Options: "aposcore", "gnina_affinity", "gnina_cnn_score", "gnina_cnn_affinity"
            save_best_structures: Whether to save the best structures (default: True)

        Returns:
            Dictionary containing scores for the best pose
        """
        logger.info("Processing scores and saving best structure...")

        # Dictionary to store best scores
        best_scores = {}
        # Dictionary to store best structures
        best_structures = {}

        # Create mapping of pocket files to ligand files
        pocket_to_ligand = {
            os.path.splitext(os.path.basename(p))[0]: l
            for p, l in zip(pocket_files, docked_ligands)
        }

        # Process each structure's scores
        for structure_id, pose_scores in scores.items():
            # Skip if no scores available
            if not pose_scores:
                logger.warning(f"No scores available for structure {structure_id}")
                continue

            # Get corresponding ligand file
            if structure_id not in pocket_to_ligand:
                logger.warning(f"No ligand file found for structure {structure_id}")
                continue

            sdf_file = pocket_to_ligand[structure_id]
            pocket_file = next(
                (
                    p
                    for p in pocket_files
                    if structure_id == os.path.splitext(os.path.basename(p))[0]
                ),
                None,
            )

            if not pocket_file:
                logger.warning(
                    f"Could not find pocket file for structure {structure_id}"
                )
                continue

            # Get protein ID from the structure ID or directory name
            protein_id = next(
                (pid for pid in protein_ids if structure_id.startswith(pid)),
                os.path.basename(os.path.dirname(sdf_file)),
            )

            # Initialize score dictionary for this protein if not exists
            if protein_id not in best_scores:
                best_scores[protein_id] = {
                    "aposcore": -float("inf"),
                    "gnina_affinity": float("inf"),
                    "gnina_cnn_score": -float("inf"),
                    "gnina_cnn_affinity": float("inf"),
                }

            # Get best ApoScore for this structure
            best_aposcore = max(pose_scores)
            # Extract GNINA scores
            gnina_scores = extract_gnina_scores_from_file(sdf_file)

            if gnina_scores:
                # Find best scores for each GNINA metric
                best_affinity = min(
                    [s["affinity"] for s in gnina_scores], default=float("inf")
                )
                best_cnn_score = max(
                    [s["cnn_score"] for s in gnina_scores], default=-float("inf")
                )
                best_cnn_affinity = max(
                    [s["cnn_affinity"] for s in gnina_scores], default=float("inf")
                )

                # Update if better than current best
                if best_aposcore > best_scores[protein_id]["aposcore"]:
                    best_scores[protein_id]["aposcore"] = best_aposcore
                    best_scores[protein_id]["gnina_affinity"] = best_affinity
                    best_scores[protein_id]["gnina_cnn_score"] = best_cnn_score
                    best_scores[protein_id]["gnina_cnn_affinity"] = best_cnn_affinity

                    # Track best structure
                    best_structures[protein_id] = {
                        "protein": pocket_file,
                        "ligand": sdf_file,
                    }
            else:
                # If no GNINA scores, just update based on ApoScore
                if best_aposcore > best_scores[protein_id]["aposcore"]:
                    best_scores[protein_id]["aposcore"] = best_aposcore
                    best_structures[protein_id] = {
                        "protein": pocket_file,
                        "ligand": sdf_file,
                    }

        # Save best structures if requested
        if save_best_structures and best_scores:
            self._save_best_structures(
                best_scores, best_structures, output_dir, rank_by
            )

        return best_scores

    def _save_best_structures(
        self,
        best_scores: Dict[str, Dict[str, float]],
        best_structures: Dict[str, Dict[str, str]],
        output_dir: str,
        rank_by: str,
    ) -> None:
        """
        Save only the best structure to output directory.

        Args:
            best_scores: Dictionary of best scores for each protein
            best_structures: Dictionary of best structures (protein and ligand paths)
            output_dir: Output directory for results
            rank_by: Which score to use for ranking ("aposcore", "gnina_affinity", etc.)
        """
        import shutil

        # Determine sort order based on rank_by parameter
        if rank_by == "aposcore":
            reverse = True  # Higher is better
            sort_key = lambda x: x[1].get("aposcore", -float("inf"))
        elif rank_by == "gnina_affinity":
            reverse = False  # Lower is better
            sort_key = lambda x: x[1].get("gnina_affinity", float("inf"))
        elif rank_by == "gnina_cnn_score":
            reverse = True  # Higher is better
            sort_key = lambda x: x[1].get("gnina_cnn_score", -float("inf"))
        elif rank_by == "gnina_cnn_affinity":
            reverse = False  # Lower is better
            sort_key = lambda x: x[1].get("gnina_cnn_affinity", float("inf"))
        else:
            logger.warning(f"Invalid rank_by value: {rank_by}. Using aposcore instead.")
            rank_by = "aposcore"
            reverse = True
            sort_key = lambda x: x[1].get("aposcore", -float("inf"))

        # Sort proteins by their scores
        sorted_items = sorted(best_scores.items(), key=sort_key, reverse=reverse)

        # Save only the top-ranked structure
        if sorted_items:
            protein_id, _ = sorted_items[0]  # Get the best protein ID
            if protein_id in best_structures:
                structure_info = best_structures[protein_id]
                protein_dir = os.path.join(output_dir, protein_id)
                os.makedirs(protein_dir, exist_ok=True)

                # Create best structure filenames using rank_by parameter
                best_protein_filename = f"{protein_id}_best_{rank_by}_protein.pdb"
                best_ligand_filename = f"{protein_id}_best_{rank_by}_ligand.sdf"

                # Copy the best structures
                try:
                    shutil.copy2(
                        structure_info["protein"],
                        os.path.join(protein_dir, best_protein_filename),
                    )
                    shutil.copy2(
                        structure_info["ligand"],
                        os.path.join(protein_dir, best_ligand_filename),
                    )
                    logger.info(
                        f"Saved best structure for {protein_id} (ranked by {rank_by})"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to save best structure for {protein_id}: {str(e)}"
                    )
