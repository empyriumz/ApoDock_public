import sys
import argparse
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
    parser = argparse.ArgumentParser(description="Docking with ApoDock")

    # Configuration file - this is the only required parameter
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
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
        # Use ResultsProcessor to rank and display the results
        pipeline.results_processor.rank_screening_results(
            best_scores, rank_by=config.rank_by, output_dir=config.output_dir
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
