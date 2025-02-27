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
        scores: np.ndarray,
        docked_ligands: List[str],
        pocket_files: List[str],
        protein_ids: List[str],
        output_dir: str,
        output_scores_file: Optional[str] = None,
        rank_by: str = "aposcore",
        save_best_structures: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Process scores for screening mode and save best structures.

        Args:
            scores: Array of scores from get_mdn_score
            docked_ligands: List of paths to docked ligand files
            pocket_files: List of paths to protein pocket files used for docking
            protein_ids: List of protein IDs
            output_dir: Output directory for results
            output_scores_file: Path to save scores (default: None)
            rank_by: Which score to use for ranking (default: "aposcore")
                Options: "aposcore", "gnina_affinity", "gnina_cnn_score", "gnina_cnn_affinity"
            save_best_structures: Whether to save the best structures (default: True)

        Returns:
            Dictionary mapping protein IDs to their best scores
        """
        logger.info(f"Processing {len(scores)} scores from screening")

        # Dictionary to store best scores for each protein
        best_scores = {}
        # Dictionary to store best structures for each protein
        best_structures = {}

        # Process all scores at once - map scores to proteins and track best structures
        score_index = 0
        for i, (sdf_file, pocket_file) in enumerate(zip(docked_ligands, pocket_files)):
            # Extract protein ID
            protein_id = (
                protein_ids[i]
                if i < len(protein_ids)
                else os.path.basename(os.path.dirname(sdf_file))
            )

            # Read molecules only once
            mols, _ = read_sdf_file(sdf_file, save_mols=True)
            if mols is None or len(mols) == 0:
                logger.warning(f"No valid molecules found in {sdf_file}")
                continue

            num_mols = len(mols)

            # Get scores for this file
            if score_index + num_mols <= len(scores):
                file_scores = scores[score_index : score_index + num_mols]
                score_index += num_mols
            else:
                # Handle case where we don't have enough scores
                remaining = len(scores) - score_index
                file_scores = scores[score_index:] if remaining > 0 else []
                score_index = len(scores)
                logger.warning(
                    f"Not enough scores for file {sdf_file}: needed {num_mols}, got {len(file_scores)}"
                )

            if len(file_scores) == 0:
                continue

            # Get best ApoScore
            best_aposcore_idx = np.argmax(file_scores)
            best_aposcore = float(file_scores[best_aposcore_idx])

            # Initialize score dictionary for this protein
            if protein_id not in best_scores:
                best_scores[protein_id] = {
                    "aposcore": -float("inf"),
                    "gnina_affinity": float("inf"),
                    "gnina_cnn_score": -float("inf"),
                    "gnina_cnn_affinity": float("inf"),
                }

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
                best_cnn_affinity = min(
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

        # Save results to CSV file if requested
        sorted_items = None
        if output_scores_file:
            sorted_items = self._save_scores_to_csv(
                best_scores, output_dir, output_scores_file, rank_by
            )

        # Save best structures if requested
        if save_best_structures and best_scores:
            self._save_best_structures(
                best_scores, best_structures, output_dir, rank_by
            )

        return best_scores

    def _save_scores_to_csv(
        self,
        best_scores: Dict[str, Dict[str, float]],
        output_dir: str,
        output_scores_file: str,
        rank_by: str = "aposcore",
    ) -> List[Tuple[str, Dict[str, float]]]:
        """
        Save scores to a CSV file and return sorted items.

        Args:
            best_scores: Dictionary mapping protein IDs to score dictionaries
            output_dir: Output directory for results
            output_scores_file: Filename for scores CSV
            rank_by: Which score to use for ranking

        Returns:
            List of sorted (protein_id, score_dict) tuples
        """
        # Set default output filename if none provided
        if not output_scores_file.endswith(".csv"):
            output_scores_file = "pocket_scores.csv"
        scores_path = os.path.join(output_dir, output_scores_file)

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
            # Default to ApoScore if invalid rank_by
            logger.warning(
                f"Invalid rank_by parameter: {rank_by}. Using aposcore instead."
            )
            rank_by = "aposcore"
            reverse = True
            sort_key = lambda x: x[1].get("aposcore", -float("inf"))

        # Sort the results
        sorted_items = sorted(best_scores.items(), key=sort_key, reverse=reverse)

        with open(scores_path, "w") as f:
            # Write header with all score types
            f.write(
                "Rank,Protein_ID,ApoScore,GNINA_Affinity,GNINA_CNN_Score,GNINA_CNN_Affinity\n"
            )

            # Write sorted results with rank
            for rank, (protein_id, score_dict) in enumerate(sorted_items, 1):
                # Get values with defaults for missing scores
                aposcore = score_dict.get("aposcore", "N/A")
                gnina_affinity = score_dict.get("gnina_affinity", "N/A")
                gnina_cnn_score = score_dict.get("gnina_cnn_score", "N/A")
                gnina_cnn_affinity = score_dict.get("gnina_cnn_affinity", "N/A")

                # Write the scores
                f.write(
                    f"{rank},{protein_id},{aposcore},{gnina_affinity},{gnina_cnn_score},{gnina_cnn_affinity}\n"
                )

        logger.info(f"Saved best scores to {scores_path} (ranked by {rank_by})")

        return sorted_items

    def _save_best_structures(
        self,
        best_scores: Dict[str, Dict[str, float]],
        best_structures: Dict[str, Dict[str, str]],
        output_dir: str,
        rank_by: str,
    ) -> None:
        """
        Save best structures to output directory.

        This helper method handles saving the best structures to a permanent location.
        For each base protein ID, it selects the best variant based on the ranking criterion.

        Args:
            best_scores: Dictionary of best scores
            best_structures: Dictionary of best structures (protein and ligand paths)
            output_dir: Output directory for results
            rank_by: Which score to use for ranking
        """
        import shutil

        # Create a sort key function based on the rank_by parameter
        if rank_by == "aposcore":
            # Higher ApoScore is better
            sort_key = lambda x: best_scores[x].get("aposcore", -float("inf"))
        elif rank_by == "gnina_affinity":
            # Lower affinity is better
            sort_key = lambda x: best_scores[x].get("gnina_affinity", float("inf")) * -1
        elif rank_by == "gnina_cnn_score":
            # Higher CNN score is better
            sort_key = lambda x: best_scores[x].get("gnina_cnn_score", -float("inf"))
        elif rank_by == "gnina_cnn_affinity":
            # Lower CNN affinity is better
            sort_key = (
                lambda x: best_scores[x].get("gnina_cnn_affinity", float("inf")) * -1
            )
        else:
            # Default to ApoScore
            sort_key = lambda x: best_scores[x].get("aposcore", -float("inf"))

        # Extract base proteins (without _pack_ suffix) and keep only best variant
        base_protein_ids = {}
        for protein_id in best_structures.keys():
            # Extract base ID (removing _pack_XX suffix if present)
            base_id = (
                protein_id.split("_pack_")[0] if "_pack_" in protein_id else protein_id
            )

            # For each base ID, keep track of its best variant
            current_variant = base_protein_ids.get(base_id)
            if current_variant is None or sort_key(protein_id) > sort_key(
                current_variant
            ):
                base_protein_ids[base_id] = protein_id

        logger.info(
            f"Saving best variant for each base protein ({len(base_protein_ids)} proteins)"
        )

        # Save best structures for each base protein
        for base_id, best_variant_id in base_protein_ids.items():
            best_structure = best_structures[best_variant_id]
            protein_dir = os.path.join(output_dir, base_id)

            # Ensure directory exists
            os.makedirs(protein_dir, exist_ok=True)

            # Create best structure filenames
            best_protein_filename = f"{base_id}_best_{rank_by}_protein.pdb"
            best_ligand_filename = f"{base_id}_best_{rank_by}_ligand.sdf"

            # Copy the best structures
            try:
                shutil.copy2(
                    best_structure["protein"],
                    os.path.join(protein_dir, best_protein_filename),
                )
                shutil.copy2(
                    best_structure["ligand"],
                    os.path.join(protein_dir, best_ligand_filename),
                )
                logger.info(f"Saved best structures for {base_id}")
            except Exception as e:
                logger.warning(
                    f"Failed to save best structures for {base_id}: {str(e)}"
                )

    def process(self, scored_poses: List[tuple], output_dir: str) -> List[str]:
        """
        Process scored poses and rank them.

        Args:
            scored_poses: List of tuples containing (docked_sdf_path, scores) or directly an array of scores
            output_dir: Output directory for processed results

        Returns:
            List of paths to the ranked pose files
        """
        result_files = []

        # Handle direct scores array from get_mdn_score
        if isinstance(scored_poses, np.ndarray):
            logger.warning(
                "Received raw scores array without file paths, cannot process results"
            )
            return result_files

        # Check if we have paired data (docked_sdf, scores) or just a list of files followed by separate scores
        if all(isinstance(item, tuple) and len(item) == 2 for item in scored_poses):
            # We have paired data - extract files and scores separately
            docked_sdfs = [item[0] for item in scored_poses]
            all_scores = []
            for _, score in scored_poses:
                if isinstance(score, (list, np.ndarray)):
                    all_scores.extend(
                        score if isinstance(score, list) else score.tolist()
                    )
                else:
                    all_scores.append(score)

            # Group the SDF files by directory (to handle different ligand/protein pairs)
            grouped_sdfs = {}
            for sdf_file in docked_sdfs:
                dir_key = os.path.dirname(sdf_file)
                if dir_key not in grouped_sdfs:
                    grouped_sdfs[dir_key] = []
                grouped_sdfs[dir_key].append(sdf_file)

            # Process each group
            for dir_key, sdf_files in grouped_sdfs.items():
                # Determine if packing was used based on file name pattern
                use_packing = any(
                    "Packed" in os.path.basename(f) or "_pack_" in os.path.basename(f)
                    for f in sdf_files
                )

                # Get the docking program name from the file path
                docking_program = "smina"  # Default
                for f in sdf_files:
                    if "gnina" in f:
                        docking_program = "gnina"
                        break

                # Count total molecules to calculate how many scores we need
                total_mols = 0
                mol_counts = []
                for sdf in sdf_files:
                    mols, _ = read_sdf_file(sdf, save_mols=True)
                    if mols is None:
                        mol_counts.append(0)
                        continue
                    mol_counts.append(len(mols))
                    total_mols += len(mols)

                # If total molecules doesn't match available scores, log warning and continue
                if total_mols > len(all_scores):
                    logger.warning(
                        f"Not enough scores ({len(all_scores)}) for all molecules ({total_mols})"
                    )
                    continue

                # Take the appropriate number of scores for this group's molecules
                group_scores = all_scores[:total_mols]
                all_scores = all_scores[total_mols:]  # Remove used scores

                # Rank the results
                ranked_file = self.rank_results(
                    sdf_files, group_scores, use_packing, docking_program
                )

                if ranked_file:
                    result_files.append(ranked_file)
        else:
            # Legacy format processing - keep for backward compatibility
            grouped_poses = {}
            for pose_info in scored_poses:
                if len(pose_info) != 2:
                    logger.warning(f"Unexpected pose info format: {pose_info}")
                    continue

                docked_sdf, scores = pose_info
                # Use the directory as a key to group related poses
                dir_key = os.path.dirname(docked_sdf)
                if dir_key not in grouped_poses:
                    grouped_poses[dir_key] = []
                grouped_poses[dir_key].append((docked_sdf, scores))

            # Process each group of poses
            for dir_key, pose_group in grouped_poses.items():
                sdf_files = [p[0] for p in pose_group]
                # Flatten the scores if they're lists
                all_scores = []
                for _, score_list in pose_group:
                    if isinstance(score_list, list):
                        all_scores.extend(score_list)
                    elif isinstance(score_list, (float, int)):
                        all_scores.append(score_list)
                    elif isinstance(score_list, np.ndarray):
                        all_scores.extend(score_list.tolist())
                    else:
                        logger.warning(f"Unexpected score type: {type(score_list)}")

                # Determine if packing was used based on file name pattern
                use_packing = any(
                    "Packed" in os.path.basename(f) or "_pack_" in os.path.basename(f)
                    for f in sdf_files
                )

                # Get the docking program name from the file path
                docking_program = "smina"  # Default
                for f in sdf_files:
                    if "gnina" in f:
                        docking_program = "gnina"
                        break

                # Rank the results
                ranked_file = self.rank_results(
                    sdf_files, all_scores, use_packing, docking_program
                )

                if ranked_file:
                    result_files.append(ranked_file)

        return result_files
