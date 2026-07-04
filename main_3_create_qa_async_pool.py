import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from commons.util_settings import load_settings
from commons.utils_msg import msg_debug, msg_error, msg_success
from main_3_create_qa import collect_source_files, get_parent_book_name, load_json_entries
from pipelines.create_qa_model_async_pool import AsyncQAPipeline


def build_text_jobs(pipeline: AsyncQAPipeline, text_files: List[Path], start_index: int) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for index, file_path in enumerate(text_files, start=1):
        if index < start_index:
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            print(msg_error(f"Failed to read {file_path}: {exc}"))
            continue
        if not text.strip():
            print(msg_debug(f"Skipping empty file: {file_path.name}"))
            continue
        parent = get_parent_book_name(file_path)
        output_jsonl = pipeline.output_dir / f"{parent}.jsonl"
        jobs.append(
            {
                "text": text,
                "source": str(file_path),
                "source_files": [str(file_path.resolve())],
                "source_key": f"text:{file_path.resolve()}",
                "output_jsonl": output_jsonl,
                "status_jsonl": pipeline.tmp_dir / f"{parent}.status.jsonl",
                "failure_jsonl": pipeline.tmp_dir / f"{parent}.failures.jsonl",
            }
        )
    return jobs


def build_json_jobs(
    pipeline: AsyncQAPipeline,
    json_files: List[Path],
    target_key: str,
    start_index: int,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    for file_path in json_files:
        entries = load_json_entries(file_path)
        output_jsonl = pipeline.output_dir / f"{file_path.stem}.jsonl"
        for index, entry in enumerate(entries, start=1):
            if index < start_index:
                continue
            value = entry.get(target_key)
            if not value or not isinstance(value, str):
                print(msg_debug(f"Entry missing target key {target_key} in {file_path.name}: {entry}"))
                continue
            stable_id = entry.get("id") or entry.get("page") or index
            jobs.append(
                {
                    "text": value,
                    "source": str(file_path),
                    "source_files": [str(file_path.resolve())],
                    "source_key": f"json:{file_path.resolve()}:{stable_id}",
                    "output_jsonl": output_jsonl,
                    "status_jsonl": pipeline.tmp_dir / f"{file_path.stem}.status.jsonl",
                    "failure_jsonl": pipeline.tmp_dir / f"{file_path.stem}.failures.jsonl",
                }
            )
    return jobs


async def async_main(settings_path: str | None, source_path: str | None, target_key: str | None, start_index: int) -> None:
    if settings_path is None:
        print(msg_error("settings_path is required."), file=sys.stderr)
        sys.exit(1)
    if source_path is None:
        print(msg_error("source path is required."), file=sys.stderr)
        sys.exit(1)

    settings = load_settings(Path(settings_path))
    source = Path(source_path).expanduser().resolve()
    text_files, json_files = collect_source_files(source)
    if json_files and not target_key:
        print(msg_error("target_key is required when processing JSON files."), file=sys.stderr)
        sys.exit(1)

    async with AsyncQAPipeline(settings) as pipeline:
        jobs = build_text_jobs(pipeline, text_files, start_index)
        if target_key:
            jobs.extend(build_json_jobs(pipeline, json_files, target_key, start_index))
        await pipeline.run_jobs(jobs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Q&A with an asyncio worker pool.")
    parser.add_argument("-p", "--settings_path", nargs="?", default="./yamls/create_qa_settings.yaml")
    parser.add_argument("-s", "--source", nargs="?", default=None)
    parser.add_argument("-t", "--target_key", type=str, default=None)
    parser.add_argument("-i", "--start_index", type=int, default=0)
    args = parser.parse_args()
    asyncio.run(async_main(args.settings_path, args.source, args.target_key, args.start_index))


if __name__ == "__main__":
    print(msg_success("Async Q&A Worker Pool Started"))
    main()
    print(msg_success("Async Q&A Worker Pool Completed"))
