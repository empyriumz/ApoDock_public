import os
import shutil
from typing import List, Dict, Tuple
from rdkit import Chem
from apodock.utils import logger, extract_gnina_scores_from_file


class ResultsProcessor:
    """Process and rank docking results."""

    def __init__(self, top_k: int = 10):
        """
        Initialize the results processor.

        Args:
            top_k: Number of top poses to keep in the final output
        """
        self.top_k = top_k

    def process_screening_results(
        self,
        scores: Dict[str, List[float]],
        docked_ligands: List[str],
        pocket_files: List[str],
        output_dir: str,
        rank_by: str = "aposcore",
        save_best_structures: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Process scores and save only the best structure and scores.

        Args:
            scores: Dictionary mapping structure IDs (e.g., '1a28_seed0_pocket_backbone_pack_25')
                   to lists of pose scores from GNINA docking
            docked_ligands: List of paths to docked ligand files
            pocket_files: List of paths to protein pocket files used for docking
            output_dir: Output directory for results
            rank_by: Which score to use for ranking (default: "aposcore")
            save_best_structures: Whether to save the best structures (default: True)

        Returns:
            Dictionary containing scores for the best pose per protein
        """
        # Dictionary to store best scores and structures per protein
        best_results = {}

        # Process each packed structure's scores
        for structure_id, pose_scores in scores.items():
            if not pose_scores:
                continue

            # Get base protein name (e.g., '1a28_seed0_pocket_backbone' from '1a28_seed0_pocket_backbone_pack_25')
            base_name = structure_id.split("_pack_")[0]

            # Find corresponding files
            try:
                sdf_idx = next(
                    i for i, path in enumerate(docked_ligands) if structure_id in path
                )
                pocket_idx = next(
                    i for i, path in enumerate(pocket_files) if structure_id in path
                )
            except StopIteration:
                continue

            # Get best score for this structure
            best_pose_score = max(pose_scores)

            # Get GNINA scores for additional metrics
            gnina_scores = extract_gnina_scores_from_file(docked_ligands[sdf_idx])

            # Initialize or update best results for this protein
            if (
                base_name not in best_results
                or best_pose_score > best_results[base_name]["scores"]["aposcore"]
            ):
                best_results[base_name] = {
                    "scores": {
                        "aposcore": best_pose_score,
                        "gnina_affinity": (
                            min(
                                [s["affinity"] for s in gnina_scores],
                                default=float("inf"),
                            )
                            if gnina_scores
                            else float("inf")
                        ),
                        "gnina_cnn_score": (
                            max(
                                [s["cnn_score"] for s in gnina_scores],
                                default=-float("inf"),
                            )
                            if gnina_scores
                            else -float("inf")
                        ),
                        "gnina_cnn_affinity": (
                            min(
                                [s["cnn_affinity"] for s in gnina_scores],
                                default=float("inf"),
                            )
                            if gnina_scores
                            else float("inf")
                        ),
                    },
                    "files": {
                        "protein": pocket_files[pocket_idx],
                        "ligand": docked_ligands[sdf_idx],
                    },
                }

        # Save best structures if requested
        if save_best_structures and best_results:
            self._save_best_structures(best_results, output_dir, rank_by)

        return {name: result["scores"] for name, result in best_results.items()}

    def _save_best_structures(
        self,
        best_results: Dict[str, Dict],
        output_dir: str,
        rank_by: str,
    ) -> None:
        """
        Save the best structure for each protein to the output directory.

        Args:
            best_results: Dictionary containing best scores and file paths for each protein
            output_dir: Output directory for results
            rank_by: Which score to use for ranking
        """

        for base_name, result in best_results.items():
            # Create protein directory
            protein_dir = os.path.join(output_dir, base_name)
            os.makedirs(protein_dir, exist_ok=True)

            # Copy protein structure
            try:
                shutil.copy2(
                    result["files"]["protein"],
                    os.path.join(protein_dir, f"best_{rank_by}_protein.pdb"),
                )

                # For ligand, read the SDF, add the ApoScore, and write it back
                ligand_file = result["files"]["ligand"]
                output_ligand_file = os.path.join(
                    protein_dir, f"best_{rank_by}_ligand.sdf"
                )

                # Read the molecule from the SDF file
                try:
                    suppl = Chem.SDMolSupplier(ligand_file, sanitize=True)
                    mols = [mol for mol in suppl if mol is not None]

                    if mols:
                        # Get the first molecule (best pose)
                        mol = mols[0]

                        # Add ApoScore as a property
                        aposcore = result["scores"]["aposcore"]
                        mol.SetProp("ApoScore", f"{aposcore:.4f}")

                        # Add other scores as properties
                        if "gnina_affinity" in result["scores"] and result["scores"][
                            "gnina_affinity"
                        ] != float("inf"):
                            mol.SetProp(
                                "GNINA_Affinity",
                                f"{result['scores']['gnina_affinity']:.4f}",
                            )
                        if "gnina_cnn_score" in result["scores"] and result["scores"][
                            "gnina_cnn_score"
                        ] != -float("inf"):
                            mol.SetProp(
                                "GNINA_CNN_Score",
                                f"{result['scores']['gnina_cnn_score']:.4f}",
                            )
                        if "gnina_cnn_affinity" in result["scores"] and result[
                            "scores"
                        ]["gnina_cnn_affinity"] != float("inf"):
                            mol.SetProp(
                                "GNINA_CNN_Affinity",
                                f"{result['scores']['gnina_cnn_affinity']:.4f}",
                            )

                        # Write the molecule to the output file
                        with Chem.SDWriter(output_ligand_file) as writer:
                            writer.write(mol)

                        logger.info(
                            f"Saved best structure for {base_name} with ApoScore {aposcore:.4f}"
                        )
                    else:
                        # If reading fails, fall back to simple copy
                        logger.warning(
                            f"Could not read molecule from {ligand_file}, falling back to simple copy"
                        )
                        shutil.copy2(ligand_file, output_ligand_file)
                except Exception as e:
                    logger.warning(
                        f"Error adding ApoScore to molecule: {str(e)}, falling back to simple copy"
                    )
                    shutil.copy2(ligand_file, output_ligand_file)

            except Exception as e:
                logger.warning(
                    f"Failed to save best structure for {base_name}: {str(e)}"
                )

    def rank_screening_results(
        self,
        best_scores: Dict[str, Dict[str, float]],
        rank_by: str = "aposcore",
        output_dir: str = None,
    ) -> List[Tuple[str, Dict[str, float], float]]:
        """
        Rank screening results based on the specified score.

        Args:
            best_scores: Dictionary mapping protein IDs to their score dictionaries
            rank_by: Which score to use for ranking (default: "aposcore")
            output_dir: Output directory for results (for logging file locations)

        Returns:
            List of tuples containing (protein_id, score_dict, rank_score) sorted by rank_score
        """
        # Sort the results based on the selected ranking score
        sorted_results = []

        for protein_id, score_dict in best_scores.items():
            # Get the ranking score with appropriate default
            if rank_by == "aposcore":
                rank_score = score_dict.get("aposcore", -float("inf"))
                reverse = True  # Higher is better
            elif rank_by == "gnina_affinity":
                rank_score = score_dict.get("gnina_affinity", float("inf"))
                reverse = False  # Lower is better
            elif rank_by == "gnina_cnn_score":
                rank_score = score_dict.get("gnina_cnn_score", -float("inf"))
                reverse = True  # Higher is better
            elif rank_by == "gnina_cnn_affinity":
                rank_score = score_dict.get("gnina_cnn_affinity", -float("inf"))
                reverse = True  # Higher is better
            else:
                logger.warning(
                    f"Unknown ranking criterion: {rank_by}, using aposcore instead"
                )
                rank_score = score_dict.get("aposcore", -float("inf"))
                reverse = True  # Higher is better

            # Skip entries with N/A for the ranking score
            if rank_score in [float("inf"), -float("inf")]:
                continue

            sorted_results.append((protein_id, score_dict, rank_score))

        # Sort the results
        sorted_results.sort(key=lambda x: x[2], reverse=reverse)

        # Log the results
        logger.info(
            f"Results ranked by {rank_by} ({'higher is better' if reverse else 'lower is better'}):"
        )

        for rank, (protein_id, score_dict, rank_score) in enumerate(sorted_results, 1):
            aposcore = score_dict.get("aposcore", "N/A")
            gnina_affinity = score_dict.get("gnina_affinity", "N/A")
            gnina_cnn_score = score_dict.get("gnina_cnn_score", "N/A")
            gnina_cnn_affinity = score_dict.get("gnina_cnn_affinity", "N/A")

            logger.info(f"  Rank {rank}: {protein_id}")
            logger.info(
                f"    ApoScore: {aposcore:.2f}"
                if aposcore != "N/A"
                else f"    ApoScore: N/A"
            )
            logger.info(
                f"    GNINA Affinity: {gnina_affinity:.2f}"
                if gnina_affinity != "N/A"
                else f"    GNINA Affinity: N/A"
            )
            logger.info(
                f"    GNINA CNN Score: {gnina_cnn_score:.2f}"
                if gnina_cnn_score != "N/A"
                else f"    GNINA CNN Score: N/A"
            )
            logger.info(
                f"    GNINA CNN Affinity: {gnina_cnn_affinity:.2f}"
                if gnina_cnn_affinity != "N/A"
                else f"    GNINA CNN Affinity: N/A"
            )

            # Log the location of best structure files if output_dir is provided
            if output_dir:
                protein_dir = os.path.join(output_dir, protein_id)
                best_protein_filename = f"{protein_id}_best_{rank_by}_protein.pdb"
                best_ligand_filename = f"{protein_id}_best_{rank_by}_ligand.sdf"

                if os.path.exists(os.path.join(protein_dir, best_protein_filename)):
                    logger.info(
                        f"    Best protein structure: {protein_dir}/{best_protein_filename}"
                    )
                    logger.info(
                        f"    Best ligand structure: {protein_dir}/{best_ligand_filename}"
                    )

        return sorted_results
