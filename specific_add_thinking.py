#!/usr/bin/env python3

import argparse
import json
import sys
import uuid
import time
from typing import Dict, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import yaml
import tqdm
from openai import OpenAI

from src.utils_print import print_info, print_error, print_success, print_debug, Colors
from src.dataloader import DataLoader


def msg_info(msg):
    return(f"{Colors.BLUE}{Colors.BOLD}💡 [INFO]{Colors.RESET} {Colors.CYAN}{msg}{Colors.RESET}")

def msg_error(msg):
    return(f"{Colors.RED}{Colors.BOLD}❌ [ERROR]{Colors.RESET} {Colors.RED}{msg}{Colors.RESET}")

def msg_debug(msg):
    return(f"{Colors.YELLOW}🔍 [DEBUG]{Colors.RESET} {Colors.GRAY}{msg}{Colors.RESET}")

def msg_success(msg):
    return(f"{Colors.GREEN}{Colors.BOLD}✅ [SUCCESS]{Colors.RESET} {Colors.GREEN}{msg}{Colors.RESET}")

def load_prompts(settings):
    # print('=' * 30)
    prompt_pathes = settings.get("prompts", None)
    create_prompt, refine_prompt = None, None
    if prompt_pathes is None:
        print_debug(f"Prompt paths in settings: {prompt_pathes}")
        sys.exit(1)
    # print_debug(Path(prompt_pathes.get('create_thinking', None)).expanduser().resolve())
    try:
        with open(Path(prompt_pathes.get('create_thinking', None)).expanduser().resolve(), "r", encoding="utf-8") as f:
            create_prompt = f.read()
            # print_debug(f"Loaded create_thinking prompt: {create_prompt[:100]}...")  # プロンプトの最初の100文字を表示
    except Exception as e:
        print_error(f"No create_thinking prompt, Create_thinking prompt must be provided. Exiting: {e}")
    try:
        with open(Path(prompt_pathes.get('refine_thinking', None)).expanduser().resolve(), "r", encoding="utf-8") as f:
            refine_prompt = f.read()
            # print_debug(f"Loaded refine_thinking prompt: {refine_prompt[:100]}...")  # プロンプトの最初の100文字を表示
    except Exception as e:
        print_error(f"Failed to load refine_thinking prompt from {prompt_pathes.get('refine_thinking', None)}: {e}")
    return create_prompt, refine_prompt

def preprocess_prompt(qa_data, prompt, reference_path, input_file_stem):
    # プロンプトを作る。
    # リファレンスのファイルも読み取ってプロンプトに入れる。
    # 例: batched_prompts = [create_prompt.format(input=item["input"], reference=read_reference(item["reference_file"])) for item in batch]
    batched_prompts = []
    for item in qa_data:
        # ここでitemから必要な情報を抽出してプロンプトを作成する処理を実装
        source_file = item.get("metadata").get("source_topic_md", "")
        source_file_path = Path(reference_path) / input_file_stem.replace('_qa', '_topics') / source_file
        try:
            with open(source_file_path, "r", encoding="utf-8") as f:
                source_data = f.read()
                # print_debug(f"{source_file}: {source_data[:100]}...")  # ソースデータの最初の100文字を表示
                # print_debug(f"Loaded source data from {source_file_path}")
        except Exception as e:
            print_error(f"Failed to load source data from {source_file_path}: {e}")
            source_data = ""
        
        # print_debug(f"item: {item}")  # itemの内容を表示して内容を確認
        
            # 実際にバッチ化する処理を実装

        if item.get("thinking", "") != "": # thinkingがすでにある場合はrefine promptを使用
            formatted_prompt = prompt.format(
                question=item.get("question", ""), 
                answer=item.get("answer", ""), 
                thinking=item.get("thinking", ""), 
                # reference=source_data # refine promptではreferenceは使用しない想定
                )
        else: # thinkingがない場合はcreate promptを使用
            formatted_prompt = prompt.format(
                question=item.get("question", ""), 
                answer=item.get("answer", ""), 
                reference=source_data
                )
        
        batched_prompts.append(formatted_prompt)
        # print('=' * 30)
        # print_debug(f"Batched prompts for item: {batched_prompts[:1]}...")  # 最初の1つのプロンプトを表示して内容を確認
    return batched_prompts

#


def _infer_text(prompt: str, settings) -> str:
    """単一のプロンプトに対して推論を実行します。
    
    Args:
        prompt: 推論に使用するプロンプト文字列。
    
    Returns:
        モデルの推論結果のテキスト。
    """
    client = OpenAI(
        api_key=settings.get("openrouter_api_key"),
        base_url=settings.get("openrouter_server_url"),
        timeout=settings.get("timeout", 30)
    )

    for _ in range(settings.get("max_retries", 3)):
        try:
            response = client.chat.completions.create(
                model=settings["openrouter_model_name"],
                messages=[{"role": "user", "content":  [{"type": "text", "text": prompt}]}],
                max_tokens=settings["infer_config"].get("max_tokens", 2048),
                temperature=settings["infer_config"].get("temperature", 0),
                top_p=settings["infer_config"].get("top_p", 1.0),
            )
            # print_debug(f"Inference response: {response.choices[0].message.content.strip()[:50]}...")  # 推論結果の最初の100文字を表示して内容を確認
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error during inference: {e}. Retrying...")
            time.sleep(settings.get("wait_seconds", 1))
    raise RuntimeError("Max retries exceeded for inference.")


def _infer_texts(prompts: List[str], settings) -> List[str]:
    """複数のプロンプトに対して並列で推論を実行します。
    
    Args:
        prompts: 推論に使用するプロンプト文字列のリスト。
    
    Returns:
        各プロンプトに対するモデルの推論結果のリスト。
        空のリストが入力された場合は空のリストを返します。
    """
    if not prompts:
        return []
    # print_info("[INFO] Starting parallel inference for prompts...")
    max_workers = len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(lambda p: _infer_text(p, settings), prompts))



def main(input_path: Path, reference_path: Path, settings_path: Path, output_path: Path=None):
    # ここで思考過程を追加する処理を実装
    input_path = Path(input_path).expanduser().resolve()
    reference_path = Path(reference_path).expanduser().resolve()
    settings_path = Path(settings_path).expanduser().resolve()

    try:
        with settings_path.open("r", encoding="utf-8") as f:
            settings = yaml.safe_load(f)
            print_debug(f"Loaded settings: {settings}")
    except Exception as e:
        print_error(f"Failed to load settings from {settings_path}: {e}")
        sys.exit(1)

    # ===============================
    # プロンプトの読み込み
    # ===============================
    create_prompt, refine_prompt = load_prompts(settings)
    # print_debug(f"Create prompt: {create_prompt[:100]}...")  # プロンプトの最初の100文字を表示
    # print_debug(f"Refine prompt: {refine_prompt[:100]}...")

    # ===============================
    # ファイルの存在確認と出力ディレクトリの作成
    # ===============================

    if output_path is not None:
        output_path = Path(settings.get("output_path", output_path)).expanduser().resolve()
    if not input_path.exists():
        print_error(f"Input file does not exist: {input_path}")
        return
    if not reference_path.exists():
        print_error(f"Reference file does not exist: {reference_path}")
        return
    if output_path.exists():
        print_info(f"Output file already exists and will be overwritten: {output_path}")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path.resolve()
    print_info(f"Input path: {input_path}")
    print_info(f"Output path: {output_path}")
    print_info(f"Reference path: {reference_path}")

    input_files = sorted(list(input_path.glob("*.json")))
    reference_files = list(reference_path.glob("*"))
    print_info("=" * 30)
    print_info(f"Found {len(input_files)} input files.")
    print_info(f"Found {len(reference_files)} reference files.")
    print_info("=" * 30)

    logging_file = Path(output_path) / "files_added_thinking.txt"
    if logging_file.exists():
        with open(logging_file, "r", encoding="utf-8") as f:
            for line in f:
                print_info(f"File already processed: {line.strip()} - skipping.")
                input_files = [f for f in input_files if f.stem != line.strip()]
        print_info(f"{len(input_files)} files remaining to process after filtering.")
    else:
        print_info("No record of previously processed files found. All files will be processed.")



    # QAデータの読み込み
    for input_file in tqdm.tqdm(input_files, desc="Processing input files..."):  # 最初の2ファイルだけ処理
        tqdm.tqdm.write(msg_info(f"Processing file: {input_file.name}")) # info
        with open(input_file, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
        # print_debug(f"Batch data: {batch}")
        # print_debug(batch[0])  # 最初のアイテムを表示して内容を確認


        # =============================
        # change json structure 
        # =============================

        temp_qa_data = []
        for item in qa_data:
            for qa in item["questions"]:
                temp_qa_data.append({
                    "id": str(uuid.uuid4()),
                    "question": qa["question"],
                    "answer": qa["answer"],
                    # "thinking": qa.get("thinking", "")
                    "metadata": {
                        "source_topic_md": item["source_file"],
                        "source_book": Path(input_file).stem.replace('_qa', ''),
                    }
                })

        tqdm.tqdm.write(msg_debug(f"Original structure: {len(qa_data)}"))  # debug 最初の1つのアイテムを表示して内容を確認
        tqdm.tqdm.write(msg_debug(f"Changed structure: {len(temp_qa_data)}"))  # debug 最初の1つのアイテムを表示して内容を確認        
        qa_data = temp_qa_data
        



        tqdm.tqdm.write(msg_debug(f"QA data loaded from {input_file}: {len(qa_data)} items"))  # debug読み込んだデータの数を表示して内容を確認

        for i in tqdm.tqdm(range(settings.get('start_idx', 0), len(qa_data), settings.get('batch_size', 1)), desc="Processing batches..."   ):
            tqdm.tqdm.write(msg_info(f"Processing items {i} to {i + settings.get('batch_size', 1)}...")) # info
            batch_data = qa_data[i:i + settings.get('batch_size', 1)] # 
            if settings.get('drop_last', False) and len(batch_data) < settings.get('batch_size', 1):
                continue
            
            # =============================
            # create thinking process
            # =============================
            tqdm.tqdm.write(msg_info(f"Creating thinking process for items {i} to {i + settings.get('batch_size', 1)}...")) # info
            batched_prompts = preprocess_prompt(batch_data, create_prompt, reference_path, input_file.stem)

            # 推論
            batch = _infer_texts(batched_prompts, settings)
            # print_debug('=' * 30)

            # print_debug(f"data len and batch len: {len(batch_data)=}, {len(batch)=}, {len(batched_prompts)=}")  # バッチデータと推論結果の数を表示して内容を確認
            # print_debug(f"Batch prompts: {batched_prompts[:1]}...")  # 最初の1つのプロンプトを表示して内容を確認


            for j in range(len(batch)):
                batch_data[j]["thinking"] = batch[j]  # thinkingを追加

            # print_debug(f"Batch data after adding thinking: {batch_data[:1]}...")  # 最初の1つのバッチデータを表示して内容を確認

            # =============================
            # refine thinking process
            # =============================
            # refine_promptが提供されている場合は、さらに推論を行う処理を実装
            if refine_prompt is not None:
                tqdm.tqdm.write(msg_info(f"Refine thinking process to items {i} to {i + settings.get('batch_size', 1)}...")) # info
                batched_prompts = preprocess_prompt(batch_data, refine_prompt, reference_path, input_file.stem)
                batch = _infer_texts(batched_prompts, settings)

                # print_debug(f"Refine thinking results: {batch[:2]}...")  # 最初の2つの推論結果を表示して内容を確認
                for j in range(len(batch)):
                    batch_data[j]["thinking"] = batch[j]  # thinkingを上書きして更新
        
            # print_debug('=' * 30)
            # print_debug(f"Output thinking item: {batch_data[:1]}...")  # 最初の1つのバッチデータを表示して内容を確認

            # add messages column
            for j in range(len(batch_data)):
                batch_data[j]["messages"] = [
                    {
                        "role": "user", 
                        "content": batch_data[j]['question']},
                    {
                        "role": "assistant", 
                        "content": f'<think>{batch_data[j]["thinking"]}</think>{batch_data[j]["answer"]}'
                    },
                ]

            # =============================
            # add thinking process to qa_data and save to output_path
            # =============================
            # qa_dataに要素を追加してoutput_pathに保存する処理を実装
            output_file = output_path / f"{input_file.stem}.jsonl"
            tqdm.tqdm.write(msg_info(f"Output file path: {output_file}")) # info
            with open(output_file, "a", encoding="utf-8") as f:
                for item in batch_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print()

        # バッチごとの処理が終わった後に、処理したファイル名をfiles_added_thinking.txtに追記する処理を実装
        with open(output_path / "files_added_thinking.txt", "a", encoding="utf-8") as f:
            f.write(f"{input_file.stem}\n")

        tqdm.tqdm.write(msg_success(f"Processed data saved to: {output_file}\n")) # success




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add thinking process to JSONL data.")
    parser.add_argument(
        "-i",
        "--input_path", 
        type=str, 
        required=True, 
        help="Path to the input JSONL file.")
    parser.add_argument(
        "-o",
        "--output_path", 
        type=str, 
        required=True, 
        help="Path to the output JSONL file.")
    parser.add_argument(
        "-r",
        "--reference", 
        type=str, 
        required=True, 
        help="Path of references for thinking process to add.")
    parser.add_argument(
        "-p",
        "--settings_path", 
        type=str, 
        required=True, 
        help="Path to the settings YAML file.")

    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    reference_path = Path(args.reference).expanduser().resolve()
    settings_path = Path(args.settings_path).expanduser().resolve()

    main(
        input_path=input_path, 
        output_path=output_path,
        reference_path=reference_path,
        settings_path=settings_path
    )

    print_info(f"Thinking process added and saved to: {output_path}")
