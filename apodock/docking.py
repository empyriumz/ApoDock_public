import sys
import argparse
import os
from apodock.config import load_config_from_yaml
from apodock.utils import (
    logger,
    ApoDockError,
    ensure_dir,
    validate_input_files,
    set_random_seed,
)
from apodock.pipeline import DockingPipeline


def parse_args():
    """
    Parse command line arguments for the docking pipeline.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Docking with ApoDock")

    # Configuration file - this is the only required parameter
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )

    # Option to display the loaded configuration
    parser.add_argument(
        "--show_config",
        action="store_true",
        help="Display the loaded configuration and exit",
    )

    return parser.parse_args()


def main():
    """
    Main function to run the ApoDock pipeline.
    """
    try:
        # Parse command line arguments
        args = parse_args()

        # Load configuration from YAML
        logger.info(f"Loading configuration from YAML file: {args.config}")
        config = load_config_from_yaml(args.config)

        # If --show_config is specified, display the configuration and exit
        if args.show_config:
            logger.info("Loaded configuration:")
            logger.info(f"  Output directory: {config.output_dir}")
            logger.info(f"  Top K poses: {config.top_k}")
            logger.info(f"  Pocket distance: {config.pocket_distance}")
            logger.info(f"  Random seed: {config.random_seed}")
            logger.info(f"  Screening mode: {config.screening_mode}")
            logger.info(f"  Output scores file: {config.output_scores_file}")
            logger.info(f"  Save poses: {config.save_poses}")
            logger.info(f"  Rank by: {config.rank_by}")
            logger.info(f"  Input files:")
            logger.info(f"    Protein: {config.input_config.protein_file}")
            logger.info(f"    Ligand: {config.input_config.ligand_file}")
            logger.info(f"    Reference ligand: {config.input_config.ref_lig_file}")
            return 0

        # Set random seed for reproducibility
        set_random_seed(config.random_seed)

        # Get input files from configuration
        if not config.input_config.ligand_file or not config.input_config.protein_file:
            raise ApoDockError(
                "Both ligand_file and protein_file must be provided in the configuration"
            )

        ligand_file = config.input_config.ligand_file
        protein_file = config.input_config.protein_file
        ref_lig_file = config.input_config.ref_lig_file

        logger.info(
            f"Using input files: Ligand={ligand_file}, Protein={protein_file}, Ref_Lig={ref_lig_file}"
        )

        # Validate inputs
        validate_input_files(ligand_file, protein_file, ref_lig_file)

        # Ensure output directory exists
        ensure_dir(config.output_dir)

        # Log configuration
        logger.info(
            "Running ApoDock with automatic detection of backbone-only structures"
        )
        logger.info(f"Output directory: {config.output_dir}")
        logger.info(f"Using random seed: {config.random_seed}")

        # Create pipeline and run in screening mode
        pipeline = DockingPipeline(config)
        best_scores = pipeline.run_screening(
            [ligand_file],
            [protein_file],
            [ref_lig_file],
            save_poses=config.save_poses,
            rank_by=config.rank_by,
        )

        # Print the best score for each protein
        logger.info("Docking results:")

        # Sort the results based on the selected ranking score
        sorted_results = []
        for protein_id, score_dict in best_scores.items():
            # Get the ranking score with appropriate default
            if config.rank_by == "aposcore":
                rank_score = score_dict.get("aposcore", -float("inf"))
                reverse = True  # Higher is better
            elif config.rank_by == "gnina_affinity":
                rank_score = score_dict.get("gnina_affinity", float("inf"))
                reverse = False  # Lower is better
            elif config.rank_by == "gnina_cnn_score":
                rank_score = score_dict.get("gnina_cnn_score", -float("inf"))
                reverse = True  # Higher is better
            elif config.rank_by == "gnina_cnn_affinity":
                rank_score = score_dict.get("gnina_cnn_affinity", -float("inf"))
                reverse = True  # Higher is better

            # Skip entries with N/A for the ranking score
            if rank_score in [float("inf"), -float("inf")]:
                continue

            sorted_results.append((protein_id, score_dict, rank_score))

        # Sort the results
        sorted_results.sort(key=lambda x: x[2], reverse=reverse)

        # Display the sorted results
        logger.info(
            f"Results ranked by {config.rank_by} ({'higher is better' if reverse else 'lower is better'}):"
        )
        for rank, (protein_id, score_dict, rank_score) in enumerate(sorted_results, 1):
            aposcore = score_dict.get("aposcore", "N/A")
            gnina_affinity = score_dict.get("gnina_affinity", "N/A")
            gnina_cnn_score = score_dict.get("gnina_cnn_score", "N/A")
            gnina_cnn_affinity = score_dict.get("gnina_cnn_affinity", "N/A")

            logger.info(f"  Rank {rank}: {protein_id}")
            logger.info(f"    ApoScore: {aposcore:.2f}")
            logger.info(f"    GNINA Affinity: {gnina_affinity:.2f}")
            logger.info(f"    GNINA CNN Score: {gnina_cnn_score:.2f}")
            logger.info(f"    GNINA CNN Affinity: {gnina_cnn_affinity:.2f}")

            # Log the location of best structure files
            protein_dir = os.path.join(config.output_dir, protein_id)
            best_protein_filename = f"{protein_id}_best_{config.rank_by}_protein.pdb"
            best_ligand_filename = f"{protein_id}_best_{config.rank_by}_ligand.sdf"

            if os.path.exists(os.path.join(protein_dir, best_protein_filename)):
                logger.info(
                    f"    Best protein structure: {protein_dir}/{best_protein_filename}"
                )
                logger.info(
                    f"    Best ligand structure: {protein_dir}/{best_ligand_filename}"
                )

        logger.info(
            f"Docking completed successfully. Results saved to: {config.output_dir}"
        )

        return 0

    except ApoDockError as e:
        logger.error(f"ApoDock Error: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        import traceback

        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
