# main_1_ocr.py

## 概要
PDF を画像化し、OCR・オブジェクト検出・内容理解・重複チェックを順に実行します。
処理結果は PDF 単位で JSONL に保存され、`main_2_sanitization.py` の入力としてそのまま利用できます。

## 実行例
```bash
python main_1_ocr.py \
  -s ./test_source/pdfs \
  -p ./yamls/ocr_settings.yaml
```

引数のデフォルト値:
- `-s/--source_pdfs_path`: `./test_pdfs/aiplan_g_20251223.pdf`
- `-p/--settings_path`: `./yamls/ocr_settings.yaml`

## 入力ファイル形式
- `-s/--source_pdfs_path`
  - 単一 PDF または PDF ディレクトリ
  - 例: `test_source/pdfs/01_shiryou2.pdf`
  - ディレクトリ指定時は直下の `*.pdf` のみ対象（再帰探索なし）

## 出力形式
- `output_path`（YAML）配下に `<pdf_stem>.jsonl` を保存します。
- 各行には主に次のキーが入ります。
  - `book`, `page`
  - `text`（OCR結果）
  - `objects_bbox`
  - `object_content`
  - `duplicate_check`
  - `ocr_generator`


## サンプル構成
- 入力サンプル: `test_source/pdfs/`
- 出力先の想定: `test_output/ocr/`

## 次工程への接続
`main_1_ocr.py` の出力 JSONL は `main_2_sanitization.py` でそのまま処理できます。

```bash
python main_2_sanitization.py \
  -s ./test_output/ocr \
  -p ./yamls/sanitization_settings_format.yaml
```

## YAML の利用方法
`yamls/ocr_settings.yaml` の主要項目:
- 推論接続
  - `openrouter`, `openrouter_api_key`, `openrouter_server_url`, `openrouter_model_name`
  - または `SERVER_URL`, `MODEL_NAME`
- 生成設定: `infer_config.max_tokens`, `temperature`, `top_p`
- 並列と再試行: `batch_size`, `max_retries`, `wait_seconds`
- プロンプト: `prompts`
  - `ocr_prompt`
  - `objects_prompt`
  - `content_understanding_prompt`
  - `duplicate_check_prompt`
  - （必要に応じて）`figure_understanding_prompt`, `table_understanding_prompt`, `photo_understanding_prompt`
- 出力先: `output_path`
