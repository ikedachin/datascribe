# Datascribe

PDF/テキスト/JSON から、以下を段階的に生成するためのパイプラインです。

1. OCR結果（JSONL）
2. サニタイズ済みテキスト（JSONL）
3. Q&Aデータ（JSONL）

`docs/` に各工程の詳細があり、この README はリポジトリ全体の実行導線をまとめたものです。

## このリポジトリの目的

このプロジェクトは、次の目的を達成するためのパイプラインです。

1. OCRを実行して、PDFなどの資料を学習可能なデータとしてデータ化する。
PDFファイルからRAGや継続事前学習に必要なデータを作ります。
2. 版権処理が必要な場合に、サニタイズ工程で必要な処理を行う。
許可をもらってocrした場合、版権に対する処理が必要です。
3. SFT用のQ&Aセットを作成する。
1や2の結果を用いて、SFT用のデータセットを作成します。
4. thinkなしのデータセットに、think部分を追加する。
think過程のない、純粋なQAデータセットにthink過程を追加します。この場合、元のデータとなる情報が必要です。（特殊事例）

## 全体フロー

通常は次の順で実行します。

1. `main_1_ocr.py`  
   PDF -> OCR JSONL（`test_output/ocr/`）
2. `main_2_sanitization.py`  
   JSON/JSONL -> sanitized JSONL（`test_output/sanitization_test/`）
3. `main_3_create_qa.py`  
   テキスト/JSON/JSONL -> QA JSONL（`test_output/test_qa/`）

既にテキスト化・JSON化されたデータがある場合は `main_2_sanitization.py` から開始できます。

## セットアップ

前提:
- Python 3.11+
- OpenRouter もしくは OpenAI互換ローカル推論サーバー

`uv`（推奨）:

```bash
uv sync
```

`venv + pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## クイックスタート

### A. 1 -> 2 -> 3 を通しで実行

```bash
# 1) OCR
python main_1_ocr.py \
  -s ./test_source/pdfs \
  -p ./yamls/ocr_settings.yaml

# 2) Sanitization（main_1出力を入力）
python main_2_sanitization.py \
  -s ./test_output/ocr \
  -p ./yamls/sanitization_settings_format.yaml

# 3) Q&A（main_2出力を入力）
# main_2でtarget_key未指定の場合は sanitized_text
python main_3_create_qa.py \
  -s ./test_output/sanitization_test \
  -t sanitized_text \
  -p ./yamls/create_qa_settings.yaml
```

### B. 既存JSON/JSONLがある場合（main_2から開始）

```bash
# 2) Sanitization（test_sourceの既存JSONLから）
python main_2_sanitization.py \
  -s ./test_source/jsonls/sample_sanitized.jsonl \
  -t original_text \
  -p ./yamls/sanitization_settings_format.yaml

# 3) Q&A（main_2で -t original_text を使ったので sanitized_original_text を指定）
python main_3_create_qa.py \
  -s ./test_output/sanitization_test \
  -t sanitized_original_text \
  -p ./yamls/create_qa_settings.yaml
```

## 入出力サンプル

入力 (`test_source/`):
- `test_source/pdfs/`（OCR入力）
- `test_source/jsons/`, `test_source/jsonls/`（Sanitization/Q&A入力）
- `test_source/texts/`, `test_source/mds/`（Q&A入力）

出力 (`test_output/`):
- `test_output/ocr/`
- `test_output/sanitization_test/`
- `test_output/test_qa/`

## 主要スクリプト

- `main_1_ocr.py`
  - PDF を画像化し、OCR・オブジェクト検出・内容理解・重複チェックを実行
  - 出力: `<pdf_stem>.jsonl`
- `main_2_sanitization.py`
  - 入力テキストを翻訳ベースでサニタイズ
  - 出力キー: `sanitized_<target_key>`, `similarity_<target_key>`, `eval_<target_key>`
  - 出力: `sanitized_<book>.jsonl`
- `main_3_create_qa.py`
  - テキスト/JSON/JSONL から Q&A を生成
  - 出力: `<parent>.jsonl`（テキスト入力時）または `<input_stem>.jsonl`（JSON入力時）
- `specific_add_thinking.py`
  - 既存QA JSONに `thinking`/`messages` を付与して JSONL 化

## 設定ファイル

各スクリプトの設定は `yamls/` を使用します。
- `yamls/ocr_settings.yaml`
- `yamls/sanitization_settings_format.yaml`
- `yamls/create_qa_settings.yaml`

主な設定項目:
- 推論接続: `openrouter` / `openrouter_api_key` / `openrouter_server_url` / `openrouter_model_name`
- ローカル推論: `SERVER_URL` / `MODEL_NAME`
- 共通: `infer_config`, `batch_size`, `max_retries`, `wait_seconds`, `output_path`


プロンプト:
各プログラムで使用するプロンプトは `./prompt/` フォルダ内にあります

## ドキュメント

詳細は `docs/` を参照してください。
- `docs/README_environment.md`
- `docs/README_main_1_ocr.md`
- `docs/README_main_2_sanitization.md`
- `docs/README_main_3_create_qa.md`
- `docs/README_specific_add_thinking.md`
