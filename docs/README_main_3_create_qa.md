# main_3_create_qa.py

## 概要
テキスト/Markdown または JSON/JSONL を入力に、Q&A を生成して JSONL に保存します。
`main_2_sanitization.py` の出力（`sanitized_*.jsonl`）も直接入力できます。

## 実行パターン
1. テキスト・Markdown から生成（`test_source/texts`, `test_source/mds`）
2. JSON/JSONL から生成（`test_source/jsons`, `test_source/jsonls`）
3. `main_2` 出力から生成（`test_output/sanitization_test`）

## 実行例

### テキスト・Markdown入力
```bash
python main_3_create_qa.py \
  -s ./test_source/mds \
  -p ./yamls/create_qa_settings.yaml
```

### JSON/JSONL入力
```bash
python main_3_create_qa.py \
  -s ./test_source/jsons \
  -t original_text \
  -p ./yamls/create_qa_settings.yaml
```

### `main_2` 出力入力
```bash
python main_3_create_qa.py \
  -s ./test_output/sanitization_test \
  -t sanitized_text \
  -p ./yamls/create_qa_settings.yaml
```

`main_2` 実行時に `-t original_text` を使った場合は、`main_3` 側は `-t sanitized_original_text` を指定してください。

## 入力ファイル形式
- 受け付け拡張子
  - テキスト系: `.md`, `.txt`
  - 構造化: `.json`, `.jsonl`
- JSON/JSONL では `-t/--target_key` が必須

入力例（`test_source/jsons/sample_documents.json`）:
```json
[
  {"book":"aaa_test","page":3,"original_text":"..."}
]
```

## 出力形式
- 出力先: YAML の `output_path`
- テキスト入力時: 親ディレクトリ名単位で `<parent>.jsonl` を作成
  - 例: `test_source/mds/aaa_test/*.md` -> `aaa_test.jsonl`
- JSON入力時: `<入力ファイルstem>.jsonl` を作成

各行の主なキー:
- `question`, `thinking`, `answer`
- `refined_thinking`, `refined_answer`
- `qa_generator`, `source_files`, `id`

## サンプル構成
- 入力サンプル:
  - `test_source/mds/aaa_test/*.md`
  - `test_source/texts/aaa_test/*.txt`
  - `test_source/jsons/sample_documents.json`
  - `test_source/jsonls/sample_sanitized.jsonl`
- 出力サンプル:
  - `test_output/test_qa/aaa_test.jsonl`
  - `test_output/test_qa/bbb_test.jsonl`
  - `test_output/test_qa/sample_documents.jsonl`

## YAML の利用方法
`yamls/create_qa_settings.yaml` の主要項目:
- 接続先:
  - OpenRouter: `openrouter`, `openrouter_api_key`, `openrouter_server_url`, `openrouter_model_name`
  - ローカル: `SERVER_URL`, `MODEL_NAME`, `NOTHINK`
- 推論設定: `infer_config`, `batch_size`, `max_retries`, `wait_seconds`
- プロンプト:
  - `question_prompt`, `thinking_prompt`, `answer_prompt`
  - `refine_thinking_prompt`, `refine_answer_prompt`
- 出力先: `output_path`
