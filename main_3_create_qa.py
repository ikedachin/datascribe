import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import tqdm

from commons.util_settings import load_settings
from commons.utils_msg import msg_debug, msg_error, msg_info, msg_success
from pipelines.create_qa_model import QAPipeline

TEXT_EXTENSIONS = {".md", ".txt"}
JSON_EXTENSIONS = {".json", ".jsonl"}


def collect_source_files(source_path: Path) -> Tuple[List[Path], List[Path]]:
    text_files: List[Path] = []
    json_files: List[Path] = []

    if source_path.is_dir():
        for candidate in sorted(source_path.rglob("*")):
            if not candidate.is_file():
                continue
            suffix = candidate.suffix.lower()
            if suffix in TEXT_EXTENSIONS:
                text_files.append(candidate)
            elif suffix in JSON_EXTENSIONS:
                json_files.append(candidate)
        return text_files, json_files

    if source_path.is_file():
        suffix = source_path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            return [source_path], []
        if suffix in JSON_EXTENSIONS:
            return [], [source_path]
        print(msg_error(f"Unsupported file type: {suffix} for {source_path}"))
        return [], []

    print(msg_error(f"Source path not found: {source_path}"))
    return [], []


def get_parent_book_name(file_path: Path) -> str:
    name = file_path.parent.name
    return name if name else file_path.stem


def load_json_entries(file_path: Path) -> List[dict]:
    entries: List[dict] = []
    suffix = file_path.suffix.lower()
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            if suffix == ".json":
                raw = json.load(f)
                if isinstance(raw, list):
                    entries = [entry for entry in raw if isinstance(entry, dict)]
                elif isinstance(raw, dict):
                    entries = [raw]
            elif suffix == ".jsonl":
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        print(msg_debug(f"Skipping invalid JSONL row in {file_path.name}: {line[:40]}"))
                        continue
                    if isinstance(obj, dict):
                        entries.append(obj)
    except Exception as exc:
        print(msg_error(f"Failed to load {file_path}: {exc}"))
    return entries


def process_text_files(
    pipeline: QAPipeline,
    text_files: List[Path],
    batch_size: int,
    start_index: int,
) -> None:
    if not text_files:
        return

    print(msg_success(f"Processing {len(text_files)} text/md file(s)."))
    output_dir = pipeline.output_dir

    parent_outputs: Dict[str, Path] = {}
    for file_path in text_files:
        parent_name = get_parent_book_name(file_path)
        parent_outputs[parent_name] = output_dir / f"{parent_name}.jsonl"
    for output_jsonl in set(parent_outputs.values()):
        if output_jsonl.exists():
            output_jsonl.unlink()

    for start in tqdm.tqdm(range(0, len(text_files), batch_size), desc="Text batches"):
        if (start + 1) < start_index:
            print(msg_info(f"Skipping text batch starting at {start + 1} for resume."))
            continue
        end = min(start + batch_size, len(text_files))
        batch_paths = text_files[start:end]
        print(msg_info(f"Text batch {start // batch_size + 1}-{end} / {len(text_files)}"))

        batch_texts: List[str] = []
        batch_sources: List[Path] = []
        batch_parents: List[str] = []
        for file_path in batch_paths:
            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception as exc:
                print(msg_error(f"Failed to read {file_path}: {exc}"))
                continue
            if not text.strip():
                print(msg_debug(f"Skipping empty file: {file_path.name}"))
                continue
            batch_texts.append(text)
            batch_sources.append(file_path)
            batch_parents.append(get_parent_book_name(file_path))

        if not batch_texts:
            continue

        results = pipeline.create_qa_batch(batch_texts, batch_size=batch_size)
        for result, source_path, parent_name in zip(results, batch_sources, batch_parents):
            result["source_files"] = [str(source_path.resolve())]
            result["id"] = str(uuid.uuid4())
            output_jsonl = parent_outputs[parent_name]
            pipeline._append_jsonl(output_jsonl, result)
            print(msg_info(f"Saved QA to: {output_jsonl}"))


def process_json_files(
    pipeline: QAPipeline,
    json_files: List[Path],
    target_key: str,
    batch_size: int,
    start_index: int,
) -> None:
    if not json_files:
        return

    print(msg_success(f"Processing {len(json_files)} JSON/JSONL file(s)."))

    for file_path in json_files:
        entries = load_json_entries(file_path)
        if not entries:
            print(msg_error(f"No entries found in {file_path}."))
            continue
        print(msg_info(f"Loaded {len(entries)} entries from {file_path.name}."))

        output_jsonl = pipeline.output_dir / f"{file_path.stem}.jsonl"
        for start in tqdm.tqdm(range(0, len(entries), batch_size), desc=f"Entries / {file_path.name}"):
            if (start + 1) < start_index:
                print(msg_info(f"Skipping {file_path.name} rows {start + 1}-{start + batch_size} for resume."))
                continue
            end = min(start + batch_size, len(entries))
            print(msg_info(f"JSON batch {start // batch_size + 1} rows {start + 1}-{end} / {len(entries)}"))

            batch_texts: List[str] = []
            for entry in entries[start:end]:
                value = entry.get(target_key)
                if not value or not isinstance(value, str):
                    print(msg_debug(f"Entry missing target key {target_key} in {file_path.name}: {entry}"))
                    continue
                batch_texts.append(value)

            if not batch_texts:
                continue

            results = pipeline.create_qa_batch(batch_texts, batch_size=batch_size)
            for result in results:
                result["source_files"] = [str(file_path.resolve())]
                result["id"] = str(uuid.uuid4())
                pipeline._append_jsonl(output_jsonl, result)

        print(msg_info(f"Saved QA to: {output_jsonl}"))


def main(
    settings_path: str | None,
    source_path: str | None,
    target_key: str | None,
    start_index: int,
) -> None:
    if settings_path is None:
        print(msg_error("settings_path is required."), file=sys.stderr)
        sys.exit(1)
    settings = load_settings(Path(settings_path))

    if source_path is None:
        print(msg_error("source path is required."), file=sys.stderr)
        sys.exit(1)
    source = Path(source_path).expanduser().resolve()

    batch_size = int(settings.get("batch_size", 1))

    text_files, json_files = collect_source_files(source)
    if not text_files and not json_files:
        print(msg_error(f"No supported files were found for {source}."), file=sys.stderr)
        sys.exit(1)

    if json_files and not target_key:
        print(msg_error("target_key is required when processing JSON files."), file=sys.stderr)
        sys.exit(1)

    pipeline = QAPipeline(settings)

    process_text_files(pipeline, text_files, batch_size, start_index)
    if target_key:
        process_json_files(pipeline, json_files, target_key, batch_size, start_index)


if __name__ == "__main__":
    print(msg_success("Q&A Creation Pipeline V2 Started"))

    parser = argparse.ArgumentParser(description="Create Q&A from text, markdown, and json files.")
    parser.add_argument(
        "-p",
        "--settings_path",
        nargs="?",
        default="./yamls/create_qa_settings.yaml",
        help="Path to the settings YAML file",
    )
    parser.add_argument(
        "-s",
        "--source",
        nargs="?",
        default=None,
        help="Path to a file or a directory containing sources",
    )
    parser.add_argument(
        "-t",
        "--target_key",
        type=str,
        default=None,
        help="Target key to extract from JSON/JSONL files",
    )
    parser.add_argument(
        "-i",
        "--start_index",
        type=int,
        default=0,
        help="Start index for resuming processing",
    )

    args = parser.parse_args()

    main(
        settings_path=args.settings_path,
        source_path=args.source,
        target_key=args.target_key,
        start_index=args.start_index,
    )

    print(msg_success("Q&A Creation Pipeline V2 Completed"))
