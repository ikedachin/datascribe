import argparse
import json
import sys
import uuid
import re
from pathlib import Path

import tqdm
import yaml

# from src.sanitize_pipeline import SanitizePipeline
from commons.util_dataloader import ListDataLoader
from commons.utils_msg import msg_info, msg_debug, msg_error, msg_success


"""
出力形式：jsonl
各行が1つのJSONオブジェクトである必要があります。
データ
{
    "book": "book_name", # ファイル名と同じ
    "page": 1,
    "original_text": "This is the content of the book.",
    "sanitized_text": "This is the sanitized content of the book.",
    "eval_score": 0.9, # llm as a judge
    "similarity": 0.8, # tfidf
}
"""


def get_files_to_sanitize(source_files: list[Path], settings: dict) -> set[str]:
    # 実行済みのファイルをスキップするためのプログラム
    print(msg_debug(f"Function start: {get_files_to_sanitize.__name__}"))
    output_dir = Path(settings["output_dir"]).expanduser().resolve()
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    sanitized_files = sorted(Path(settings["output_dir"]).glob("**/*.jsonl"))
    loaded_files = load_files(source_files) # keys = book, page, original_text

    sanitized_info = {}
    for sanitized_file in sanitized_files:
        with open(sanitized_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data['book'] not in sanitized_info:
                    sanitized_info[data['book']] = [data['page']]
                else:
                    sanitized_info[data['book']].append(data['page'])
    print(msg_debug(f"sanitized_info: {sanitized_info}"))

    output_files = []
    for loaded_file in loaded_files:
        book = loaded_file['book']
        page = loaded_file['page']
        if book in sanitized_info and page in sanitized_info[book]:
            print(msg_debug(f"Sanitized: {loaded_file}"))
        else:
            output_files.append(loaded_file)
    print(msg_debug(f"Function end: {get_files_to_sanitize.__name__}"))
    return output_files


def load_files(source_files: list[Path]) -> list[dict]:
    # print(msg_debug(source_files))

    loaded_files = []
    for file_path in sorted(source_files):
        print(msg_debug(file_path))
        book = file_path.parent.name
        match = re.search(r'\d{4}', file_path.name)
        page = match.group() if match else None
        if file_path.suffix in ['.txt', '.md', '.csv']:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                loaded_files.append({"book": book, "page": int(page), "original_text": content})
    return loaded_files




def main(args) -> None:
    # args contain settings_path source, target_keys, start_index

    # ======================================================
    # 条件設定
    # ======================================================

    # 設定値の取得
    if args.settings is None:
        print(msg_debug("settings is required."))
        return
    else:
        settings_path = Path(args.settings).expanduser().resolve()

        with open(settings_path, "r", encoding="utf-8") as f:
            settings = yaml.safe_load(f)
            print(msg_debug(f"Settings loaded: \n{settings}"))
    print(msg_debug(f"Loading settings from: {settings_path}"))

    # 引数をすべて設定に追加
    if args.extensions:
        settings["extensions"] = ["." + ext.strip() if not ext.startswith(".") else ext.strip() for ext in args.extensions.split(",")]
        print(msg_debug(f"Extensions: {settings['extensions']}"))

    # ターゲットキーの取得
    if args.target_key is not None:
        settings['target_key'] = args.target_key
        print(msg_debug(f"Target key: {args.target_key}"))

    # ソースパスの取得
    if args.source is None:
        print(msg_debug("source is required."))
        return
    else:
        # for file in Path(args.source).expanduser().resolve().glob("**/*.txt"):
        #     print(file)
        source_path = Path(args.source).expanduser().resolve()
        settings["source_path"] = source_path
        print(msg_debug(f"Source path: {source_path}"))
        if source_path.is_dir():
            source_files = []
            for file in source_path.glob("**/*"):
                if file.is_file():
                    if settings['extensions']:
                        if file.suffix in settings['extensions']:
                            source_files.append(file)
                        else:
                            print(msg_debug(f"File excluded by extension: {file}"))
                    else:
                        source_files.append(file)
            source_files = sorted(source_files)
        else:
            source_files = [source_path]


    print(msg_debug(f"Source files Nums: {len(source_files)}"))

    # アウトプットファイルから実行されていないファイルだけにする
    source_files = get_files_to_sanitize(source_files, settings)


    # ======================================================
    # パイプラインの初期化と実行
    # ======================================================

    # pipeline = SanitizePipeline(settings)
    dataloader = ListDataLoader(source_files, settings)
    for data, start_index, end_index in dataloader:

        print(msg_debug(f"batched files: {data=}, {start_index=}, {end_index=}"))

        # results = pipeline.sanitize_batch(
        #     data=data,
        # )
        # pipeline.save_results(results)



if __name__ == "__main__":
    print(msg_success("Sanitization Pipeline Started"))

    parser = argparse.ArgumentParser(description="Sanitize datas.")
    parser.add_argument(
        "-s",
        "--source",
        nargs="?",
        default=None,
        help="Path to a file or a directory include .md .json, .jsonl, .txt",
    )
    parser.add_argument(
        "-p",
        "--settings",
        nargs="?",
        default="./yamls/sanitization_settings.yaml",
        help="Path to the settings YAML file",
    )
    parser.add_argument(
        "-t",
        "--target_key",
        type=str,
        default=None,
        help="target key to extract from JSONL files",
    )
    parser.add_argument(
        "-i",
        "--start_index",
        type=int,
        default=0,
        help="Start index for resuming processing",
    )
    parser.add_argument(
        "-e",
        "--extensions",
        type=str,
        default=None,
        help="File extensions to process (e.g. .md, .json, .jsonl, .txt)",
    )
    
    args = parser.parse_args()

    main(args)

    print(msg_success("Sanitization Pipeline Completed"))
