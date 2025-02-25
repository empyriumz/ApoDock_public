import sys
import argparse

from apodock.config import load_config_from_yaml
from apodock.utils import logger, ApoDockError, ensure_dir, validate_input_files
from apodock.pipeline import DockingPipeline


def parse_args():
    """
    Parse command line arguments for the docking pipeline.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Docking with ApoDock")

    # Configuration file
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
            f"Running ApoDock with {'packing' if config.use_packing else 'direct docking'}"
        )
        logger.info(f"Output directory: {config.output_dir}")

        # Create and run pipeline
        pipeline = DockingPipeline(config)
        results = pipeline.run([ligand_file], [protein_file], [ref_lig_file])

        logger.info(
            f"Docking completed successfully. Results saved to: {', '.join(results)}"
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
