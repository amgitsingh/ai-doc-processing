import os
import json
from dotenv import load_dotenv

def load_config(config_path: str = "config/config.json"):
    load_dotenv()

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config
