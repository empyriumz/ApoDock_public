import os
from typing import List, Optional
import numpy as np
from rdkit import Chem

from apodock.utils import logger
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
