# main_2_sanitization_async_pool.py

## 概要
入力テキストを翻訳ベースでサニタイズし、評価値と類似度を付与して JSONL へ保存します。
`main_1_ocr_async_pool.py` の出力 JSONL、または既にテキスト化済みの JSON/JSONL を入力できます。

## 実行パターン
1. 通常フロー（`main_1` の出力を受ける）
2. 途中開始（既存の JSON/JSONL から `main_2` を開始）

## 実行例
### 1) `main_1` 出力を入力する
```bash
python main_2_sanitization_async_pool.py \
  -s ./test_output/ocr \
  -t final_output \
  -p ./yamls/sanitization_settings_format.yaml
```

### 2) `test_source` の既存 JSONL から開始する
```bash
python main_2_sanitization_async_pool.py \
  -s ./test_source/jsonls/sample_sanitized.jsonl \
  -t original_text \
  -p ./yamls/sanitization_settings_format.yaml
```

## 入力ファイル形式
- `-s/--source`: ファイルまたはディレクトリ
- 受け付け拡張子: `.json`, `.jsonl`（`-e` 指定時はフィルタ可能）
- 各レコードは次のいずれかのキーで本文を解釈:
  - `original_text`（優先）
  - `text`
  - `content`
- 実際にサニタイズするキーは `-t/--target_key`（未指定時は `text`）で決まります。

例（`test_source/jsonls/sample_sanitized.jsonl`）:
```json
{"book":"aaa_test","page":1,"original_text":"..."}
```

## 出力形式
- 出力先: `output_path`
- ファイル名: `sanitized_<book>.jsonl`
- Async worker pool 版のファイル配置:
  - 最終出力: `output_path` 直下の `sanitized_<book>.jsonl`（常に page 順。tmp から毎回ソートして再生成）
  - 中間出力（完了順の追記。resume 台帳）: `output_path`/`tmp`/`sanitized_<book>_tmp.jsonl`
  - 失敗ログ: `output_path`/`tmp`/`sanitized_<book>.failures.jsonl`
  - 再実行時の skip は tmp の `book` / `page` で判定し、失敗 item は自動で再処理されます。
  - 入力ディレクトリ走査では `*_tmp.jsonl` と `*.failures.jsonl` を自動除外します（OCR の tmp を誤って読まないため）。
- 各行の主なキー:
  - `book`, `page`, `original_text`
  - `sanitized_<target_key>`
  - `similarity_<target_key>`
  - `eval_<target_key>`（`eval_prompt` がある場合）
  - `generator`

## サンプル構成
- 入力サンプル:
  - `test_source/jsonls/sample_sanitized.jsonl`
  - `test_source/jsons/sample_documents.json`
- 出力サンプル:
  - `test_output/sanitization_test/sanitized_aaa_test.jsonl`
  - `test_output/sanitization_test/sanitized_bbb_test.jsonl`

## YAML の利用方法
`yamls/sanitization_settings_format.yaml` の主要項目:
- 接続先:
  - OpenRouter: `openrouter`, `openrouter_api_key`, `openrouter_server_url`, `openrouter_model_name`
  - ローカル: `SERVER_URL`, `MODEL_NAME`
- 推論設定: `infer_config`, `batch_size`, `max_retries`, `wait_seconds`
- Async 版: `max_in_flight`, `max_connections`, `max_keepalive_connections`, `connect_timeout`, `read_timeout`, `write_timeout`, `pool_timeout`, `http2`
- プロンプト（順序付き）:
  - `jp_en_prompt`, `en_jp_prompt`, `refine_prompt`, `eval_prompt`
- 出力先: `output_path`

`-t/--target_key` を指定すると、そのキー名に対して `sanitized_<target_key>` を生成します。

## 処理の構成
`main_2_sanitization_async_pool.py` は item 単位で `sanitize_one()` を実行し、保存と失敗記録を worker pool 側で処理します。

- `max_in_flight`: プログラム全体で同時に外部APIへ投げる最大リクエスト数
- 再実行時のskip: `tmp/sanitized_*_tmp.jsonl` の `book` / `page`
- 失敗記録: `tmp/sanitized_<book>.failures.jsonl`

## 次工程への接続
生成された `sanitized_*.jsonl` は `main_3_create_qa_async_pool.py` で利用できます。

```bash
python main_3_create_qa_async_pool.py \
  -s ./test_output/sanitization_test \
  -t sanitized_original_text \
  -p ./yamls/create_qa_settings.yaml
```

