import argparse
import asyncio
from pathlib import Path

import yaml

from commons.utils_msg import msg_debug, msg_error, msg_success
from main_2_sanitization import load_files
from pipelines.sanitize_pipeline_async_pool import AsyncSanitizePipeline


def collect_source_files(source: str | None, extensions: str | None) -> list[Path]:
    if source is None:
        print(msg_error("source is required."))
        return []
    source_path = Path(source).expanduser().resolve()
    allowed = None
    if extensions:
        allowed = {"." + ext.strip() if not ext.startswith(".") else ext.strip() for ext in extensions.split(",")}
    if source_path.is_dir():
        # OCR async版の中間ファイル・失敗ログは入力から除外する
        files = [
            p for p in source_path.glob("**/*")
            if p.is_file()
            and not p.name.endswith("_tmp.jsonl")
            and not p.name.endswith(".failures.jsonl")
        ]
        if allowed:
            files = [p for p in files if p.suffix in allowed]
        return sorted(files)
    return [source_path]


async def async_main(args) -> None:
    if args.settings_path is None:
        print(msg_error("settings_path is required."))
        return
    settings_path = Path(args.settings_path).expanduser().resolve()
    with open(settings_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    if args.target_key is not None:
        settings["target_key"] = args.target_key
    if args.extensions:
        settings["extensions"] = [
            "." + ext.strip() if not ext.startswith(".") else ext.strip()
            for ext in args.extensions.split(",")
        ]
    print(msg_debug(f"Settings loaded: {settings}"))

    source_files = collect_source_files(args.source, args.extensions)
    rows = load_files(source_files)
    async with AsyncSanitizePipeline(settings) as pipeline:
        await pipeline.run(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitize data with an asyncio worker pool.")
    parser.add_argument("-s", "--source", nargs="?", default=None)
    parser.add_argument("-p", "--settings_path", nargs="?", default="./yamls/sanitization_settings.yaml")
    parser.add_argument("-t", "--target_key", type=str, default=None)
    parser.add_argument("-i", "--start_index", type=int, default=0)
    parser.add_argument("-e", "--extensions", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    print(msg_success("Async Sanitization Worker Pool Started"))
    main()
    print(msg_success("Async Sanitization Worker Pool Completed"))
