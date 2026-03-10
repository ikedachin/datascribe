#!/usr/bin/env python3
import yaml
from pathlib import Path
from commons.utils_msg import msg_info, msg_error, msg_debug    


def load_settings(settings_path: Path) -> dict:
    if isinstance(settings_path, str):
        settings_path = Path(settings_path)
    if not settings_path.is_file():
        raise FileNotFoundError(msg_error(f"Settings file not found: {settings_path}"))

    with open(settings_path, "r", encoding='utf-8') as f:
        settings = yaml.safe_load(f)
        print(msg_debug(f"Loaded settings: {settings}"))

    return settings

def get_inference_config(settings: dict) -> dict:
    # READ SETTINGS
    inference_config = settings.get("infer_config", {})
    MAX_TOKENS = inference_config.get("max_tokens", 2048)
    TEMPERATURE = inference_config.get("temperature", 0)
    TOP_P = inference_config.get("top_p", 1.0)
    json_output_path = settings.get("json_output", "./outputs/json/result.json")
    # print(msg_debug(f"Loaded Inference Config: {inference_config}"))
    
    if not Path(json_output_path).exists():
        print(msg_info(f"Creating output directory: {Path(json_output_path).parent}"))
        Path(json_output_path).mkdir(parents=True, exist_ok=True)
    if settings.get("openrouter", False):
        API_KEY = settings.get("openai_api_key", "dummy")
        SERVER_URL = settings.get("oupenrouter_server_url", "https://openrouter.ai/api/v1")
        MODEL_NAME = settings.get("openrouter_model_name", None)
        print(msg_debug("Using OpenRouter settings"))
    else:
        API_KEY = "dummy"
        SERVER_URL = settings.get("SERVER_URL", "http://localhost:8000/v1")
        MODEL_NAME = settings.get("MODEL_NAME", None)
        print("Using local server settings")

    inference_config.update({
        "API_KEY": API_KEY,
        "SERVER_URL": SERVER_URL,
        "MODEL_NAME": MODEL_NAME,
        "output_path": settings.get("json_output", "./outputs/json/result.json"),
    })
    print(msg_debug(f"Inference Config: {inference_config.keys()}"))
    return inference_config
