import yaml
from typing import Any, Dict

def load_config(yaml_path: str) -> Dict[str, Any]:
    """
    Reads the YAML file and returns it entirely as a dict.
    It is NOT pre-split into General/Data/Training sections.
    """
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file '{yaml_path}' not found.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing the YAML file: {e}")

    if not isinstance(config, dict):
        logging.error("Expected a mapping (dict) in the YAML. E.g. `{ General: { … }, Data: { … }, Training: { … } }`.")
        sys.exit(1)

    return config