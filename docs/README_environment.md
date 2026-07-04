# Datascribe 環境構築

## 1. 前提
- Python 3.11 以上
- API 接続先
  - OpenRouter を使う場合: `openai_api_key` または `openrouter_api_key`
  - ローカル推論サーバーを使う場合: OpenAI互換エンドポイント (`SERVER_URL`)

## 2. セットアップ

### uv を使う場合（推奨）
```bash
cd datascribe
uv sync
```

### venv + pip を使う場合
```bash
cd datascribe
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. ディレクトリの役割
- `test_source/`: サンプル入力
- `test_output/`: サンプル出力
- `yamls/`: 各プログラムの設定ファイル
- `prompts/`: 推論プロンプト
- `docs/`: 本ドキュメント

## 3.1 実行フロー（main_1 -> main_3）
通常は次の順番で実行します。

1. `main_1_ocr_async_pool.py`: PDF -> JSONL（OCR結果）
2. `main_2_sanitization_async_pool.py`: JSON/JSONL -> サニタイズ済み JSONL
3. `main_3_create_qa_async_pool.py`: テキスト/JSON/JSONL -> Q&A JSONL

非同期 worker pool 版を使う場合は、対応する async 入口を使います。

1. `main_1_ocr_async_pool.py`: PDF -> JSONL（OCR結果）
2. `main_2_sanitization_async_pool.py`: JSON/JSONL -> サニタイズ済み JSONL
3. `main_3_create_qa_async_pool.py`: テキスト/JSON/JSONL -> Q&A JSONL

Async 版では YAML の `max_in_flight` がプログラム全体の同時APIリクエスト上限です。失敗した item は `*.failures.jsonl` に保存され、OCR / Sanitization は既存出力の `book` / `page`、Q&A は `*.status.jsonl` で再実行時のskipを行います。

`test_source` には各段階で試せる入力が入っています。
- OCR入力（PDF）: `test_source/pdfs/`
- サニタイズ入力（JSON/JSONL）: `test_source/jsons/`, `test_source/jsonls/`
- Q&A入力（テキスト/Markdown）: `test_source/texts/`, `test_source/mds/`

既にテキスト化・JSON化されたデータがある場合は、`main_2_sanitization_async_pool.py` から開始できます。

## 4. APIキーの設定
このリポジトリは各 YAML 内のキーを直接読みます。用途に合わせて以下を設定してください。
- OpenRouter利用: `openrouter: true` と APIキー（`openai_api_key` もしくは `openrouter_api_key`）
- ローカル利用: `openrouter: false` と `SERVER_URL`, `MODEL_NAME`

## 5. 動作確認の例
```bash
python main_2_sanitization_async_pool.py \
  -s ./test_source/jsonls/sample_sanitized.jsonl \
  -t original_text \
  -p ./yamls/sanitization_settings_format.yaml
```

必要に応じて YAML の `output_path` / `output_dir` などの設定ファイルを変更してください。
