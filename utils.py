import yaml
from pathlib import Path

def load_config(config_path):
    """Load additional hyperparameters from a YAML config file."""
    config_path = Path(config_path)
    if config_path.is_file():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)