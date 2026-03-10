#!/usr/bin/env python3
import yaml
from pathlib import Path
from commons.utils_msg import msg_info, msg_error, msg_debug    


def load_settings(settings_path: Path) -> dict:
    if isinstance(settings_path, str):
        settings_path = Path(settings_path)
    if not settings_path.is_file():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")
    
    with open(settings_path, "r") as f:
        settings = yaml.safe_load(f)
        print(msg_debug(f"Loaded settings: {settings}"))

    return settings


    # inference_config = settings.get("infer_config", {})
    # MAX_TOKENS = inference_config.get("max_tokens", 2048)
    # TEMPERATURE = inference_config.get("temperature", 0)
    # TOP_P = inference_config.get("top_p", 1.0)
    # json_output_path = settings.get("json_output", "./outputs/json/result.json")
