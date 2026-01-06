# main.py
import argparse
import os
import sys
import logging

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from muzero import ReinforcementLearningManager

logger = logging.getLogger(__name__)


def main():
    """Main function to initialize and run the MuZero training loop."""
    parser = argparse.ArgumentParser(description="Run MuZero training.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()
    config_path = args.config

    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at '{config_path}'")
        sys.exit(1)

    logger.info(f"Using configuration file: {config_path}")

    try:
        logger.info("Initializing ReinforcementLearningManager...")
        rl_manager = ReinforcementLearningManager(config_path=config_path)
        logger.info("Initialization complete.")

        rl_manager.run_training_loop()

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration or Initialization Error: {e}")
        sys.exit(1)
    except ImportError as e:
        logger.critical(f"Import Error: {e}. Check dependencies and project structure.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"\n--- An unexpected error occurred ---")
        logger.exception(e)
        logger.critical("------------------------------------\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
