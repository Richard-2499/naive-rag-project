from pathlib import Path
from typing import Any

import yaml

CONFIG_FILE = Path(__file__).parent / "config.yaml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict[str, Any]:
    """
    Load configuration from yaml file.
    """
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_FILE}")

    with CONFIG_FILE.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError(f"Config file is empty")

    paths = config.get("paths")
    # for key, value in paths.items():
    #     paths[key] = PROJECT_ROOT / value
    resolved_paths = {
        key: PROJECT_ROOT / value
        for key, value in paths.items()
    }
    config["paths"] = resolved_paths


    return config

if __name__ == "__main__":
    config = load_config()
    print(config)