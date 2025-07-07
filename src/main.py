import argparse
import yaml
import logging

from utils.args import load_config
from utils.fine_tuning import fine_tune
from utils.training import train
from utils.synthesization import synthesize
#from utils.synthesization_leo import synthesize_leo

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(
        description="main.py expects a path to a YAML configuration file."
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Path to the YAML file, e.g. 'config.yaml'."
    )
    args = parser.parse_args()

    logging.info(f"Loading configuration from: %s", args.config_path)
    cfg = load_config(args.config_path)

    op = None
    if "General" in cfg and isinstance(cfg["General"], dict):
        op = cfg["General"].get("operation")
    else:
        logging.error("General section not found or invalid in the configuration file.")
        raise ValueError("General section not found or invalid in the configuration file.")

    logging.info(f"Operation requested: %s", op)
    
    if op == "fine-tuning":
        logging.info("Starting fine-tuning process.")
        general_cfg = cfg.get("General", {}) if isinstance(cfg.get("General"), dict) else {}
        data_cfg = cfg.get("Data", {}) if isinstance(cfg.get("Data"), dict) else {}
        training_cfg = cfg.get("Training", {}) if isinstance(cfg.get("Training"), dict) else {}

        try:
            fine_tune(general_cfg=general_cfg, data_cfg=data_cfg, training_cfg=training_cfg)
            logging.info("Fine-tuning completed successfully.")
        except Exception as e:
            logging.exception("An error occurred during fine-tuning: %s", e)
            raise
    
    elif op == "training":
        logging.info("Starting training process.")
        general_cfg = cfg.get("General", {}) if isinstance(cfg.get("General"), dict) else {}
        data_cfg = cfg.get("Data", {}) if isinstance(cfg.get("Data"), dict) else {}
        training_cfg = cfg.get("Training", {}) if isinstance(cfg.get("Training"), dict) else {}

        try:
            train(general_cfg=general_cfg, data_cfg=data_cfg, training_cfg=training_cfg)
            logging.info("Training completed successfully.")
        except Exception as e:
            logging.exception("An error occurred during training: %s", e)
            raise

    elif op == "synthesization":
        logging.info("Starting synthesization process.")
        general_cfg = cfg.get("General", {}) if isinstance(cfg.get("General"), dict) else {}
        data_cfg = cfg.get("Data", {}) if isinstance(cfg.get("Data"), dict) else {}
        generation_cfg = cfg.get("Generation", {}) if isinstance(cfg.get("Generation"), dict) else {}

        try:
            synthesize(general_cfg=general_cfg, data_cfg=data_cfg, generation_cfg=generation_cfg)
            logging.info("Synthesization completed successfully.")
        except Exception as e:
            logging.exception("An error occurred during synthesization: %s", e)
            raise

    elif op == "prediction":
        logging.info("Starting prediction process.")
        # Prediction logic would go here
        try:
            predict()
            logging.info("Prediction completed successfully.")
        except Exception as e:
            logging.exception("An error occurred during prediction: %s", e)
            raise

    else:
        logging.error(f"Unknown operation: %s", op)
        raise ValueError(f"Unknown operation: {op}")


if __name__ == "__main__":
    main()