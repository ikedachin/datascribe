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

## 全体フロー

通常は次の順で実行します。

1. `main_1_ocr_async_pool.py`  
   PDF -> OCR JSONL（`test_output/ocr/`）
2. `main_2_sanitization_async_pool.py`  
   JSON/JSONL -> sanitized JSONL（`test_output/sanitization_test/`）
3. `main_3_create_qa_async_pool.py`  
   テキスト/JSON/JSONL -> QA JSONL（`test_output/test_qa/`）

既にテキスト化・JSON化されたデータがある場合は `main_2_sanitization_async_pool.py` から開始できます。

全スクリプトが asyncio worker pool 方式です。`asyncio.Queue` / `asyncio.create_task` / `asyncio.Semaphore` / `httpx.AsyncClient` で item 単位に処理し、各 item は同じ worker が最後の保存または失敗記録まで処理します。

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
python main_1_ocr_async_pool.py \
  -s ./test_source/pdfs \
  -p ./yamls/ocr_settings.yaml

# 2) Sanitization（main_1出力を入力。-t final_output で織り込み済みテキストを対象化）
python main_2_sanitization_async_pool.py \
  -s ./test_output/ocr \
  -t final_output \
  -p ./yamls/sanitization_settings_format.yaml

# 3) Q&A（main_2出力を入力）
python main_3_create_qa_async_pool.py \
  -s ./test_output/sanitization_test \
  -t sanitized_final_output \
  -p ./yamls/create_qa_settings.yaml
```

`max_in_flight` がプログラム全体で同時に外部 API へ投げてよい最大リクエスト数です。`batch_size` は既存互換の設定で、`max_in_flight` 未指定時の既定値として使われます。

接続設定は YAML で調整できます。

- `max_connections`
- `max_keepalive_connections`
- `connect_timeout`
- `read_timeout`
- `write_timeout`
- `pool_timeout`
- `http2`

失敗した item は処理全体を止めず、`output_path/tmp/` に `*.failures.jsonl` として保存します。最終出力は `output_path` 直下、中間ファイル（resume 台帳・失敗ログ・status）は `output_path/tmp/` に置かれます。OCR / Sanitization は tmp の中間 JSONL の `book` / `page` から完了済みを skip し、最終出力は page 順で再生成されます。Q&A は出力 JSONL の schema を変えず、`tmp/*.status.jsonl` の sidecar で完了済み source を管理します。

### C. 既存JSON/JSONLがある場合（main_2から開始）

```bash
# 2) Sanitization（test_sourceの既存JSONLから）
python main_2_sanitization_async_pool.py \
  -s ./test_source/jsonls/sample_sanitized.jsonl \
  -t original_text \
  -p ./yamls/sanitization_settings_format.yaml

# 3) Q&A（main_2で -t original_text を使ったので sanitized_original_text を指定）
python main_3_create_qa_async_pool.py \
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

- `main_1_ocr_async_pool.py`
  - PDF を画像化し、OCR・オブジェクト検出・内容理解・参照統合を実行（2 パス構成）
  - 1 パス目: `pipelines/ocr_model_worker_pool.py` で単ページ画像に対し `ocr -> objects_bbox -> object_content -> 保存` を item 単位に実行
  - 2 パス目: `pipelines/ocr_reference_integration.py` で図・表・写真の参照ページを判定（番号マッチ -> BM25）し、参照元ページの本文に内容を織り込んだ `final_output` を生成
  - 出力: `<pdf_stem>.jsonl`（最終、output_path直下・page 順）, `tmp/<pdf_stem>_tmp.jsonl`（中間）, 失敗: `tmp/<pdf_stem>.failures.jsonl`
- `main_2_sanitization_async_pool.py`
  - 入力テキストを翻訳ベースでサニタイズ
  - 出力キー: `sanitized_<target_key>`, `similarity_<target_key>`, `eval_<target_key>`
  - 出力: `sanitized_<book>.jsonl`（page 順）, 中間: `tmp/sanitized_<book>_tmp.jsonl`, 失敗: `tmp/sanitized_<book>.failures.jsonl`
- `main_3_create_qa_async_pool.py`
  - テキスト/JSON/JSONL から Q&A を生成
  - 出力: `<parent>.jsonl`（テキスト入力時）または `<input_stem>.jsonl`（JSON入力時）
  - resume 用に `tmp/*.status.jsonl` を併用、失敗: `tmp/*.failures.jsonl`

## 設定ファイル

各スクリプトの設定は `yamls/` を使用します。
- `yamls/ocr_settings.yaml`
- `yamls/sanitization_settings_format.yaml`
- `yamls/create_qa_settings.yaml`

主な設定項目:
- 推論接続: `openrouter` / `openrouter_api_key` / `openrouter_server_url` / `openrouter_model_name`
- ローカル推論: `SERVER_URL` / `MODEL_NAME`
- 共通: `infer_config`, `batch_size`, `max_retries`, `wait_seconds`, `output_path`
- Async 版: `max_in_flight`, `max_connections`, `max_keepalive_connections`, `connect_timeout`, `read_timeout`, `write_timeout`, `pool_timeout`, `http2`
- OCR async 版: `dpi`, `pdf_concurrency`, `reference_window`, `bm25_top1_ratio`


プロンプト:
各プログラムで使用するプロンプトは `./prompt/` フォルダ内にあります

## ドキュメント

詳細は `docs/` を参照してください。
- `docs/README_environment.md`
- `docs/README_main_1_ocr.md`
- `docs/README_main_2_sanitization.md`
- `docs/README_main_3_create_qa.md`
