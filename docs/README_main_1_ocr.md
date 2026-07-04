# main_1_ocr_async_pool.py

## 概要
PDF を画像化し、OCR・オブジェクト検出・内容理解・図表の参照統合を実行します（asyncio worker pool、2 パス構成）。
処理結果は PDF 単位で JSONL に保存され、`main_2_sanitization_async_pool.py` の入力としてそのまま利用できます。

## 実行例
```bash
python main_1_ocr_async_pool.py \
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
- ファイル配置:
  - 最終出力（2 パス目の参照統合結果）: `output_path` 直下の `<pdf_stem>.jsonl`（常に page 順）
  - 中間出力（1 パス目のページ単位結果。resume 台帳兼 2 パス目の入力）: `output_path`/`tmp`/`<pdf_stem>_tmp.jsonl`
  - 失敗ログ: `output_path`/`tmp`/`<pdf_stem>.failures.jsonl`
- 各行には主に次のキーが入ります。
  - `book`, `page`
  - `text`（OCR結果）
  - `objects_bbox`
  - `object_content`
  - `ocr_generator`
- 最終出力にはさらに次のキーが入ります。
  - `final_output`: 図・表・写真の内容を織り込んだ最終テキスト
  - `woven_objects`: 織り込んだ図表の出典（`source_page` / `object_index` / `method`）


## サンプル構成
- 入力サンプル: `test_source/pdfs/`
- 出力先の想定: `test_output/ocr/`

## 次工程への接続
`main_1_ocr_async_pool.py` の出力 JSONL は `main_2_sanitization_async_pool.py` でそのまま処理できます。

```bash
python main_2_sanitization_async_pool.py \
  -s ./test_output/ocr \
  -p ./yamls/sanitization_settings_format.yaml
```

## 処理の構成
`main_1_ocr_async_pool.py` は 2 パス構成です。

1 パス目（`pipelines/ocr_model_worker_pool.py`）: ページ単位（画像 1 枚）の worker pool。各 item は同じ worker が `ocr -> objects_bbox -> object_content -> 保存` まで処理し、`tmp/<pdf_stem>_tmp.jsonl` に書き込みます（OCR プロンプトは単ページ用 `ocr_single_page.txt` を使用）。

2 パス目（`pipelines/ocr_reference_integration.py`）: PDF の全ページ完了後、図・表・写真ごとに「前後 `reference_window` ページ以内で参照しているページ」を判定します。

- 判定方法: キャプション番号マッチ（「図3」「表2-1」等の正規表現）を優先し、番号が取れない図表は BM25（文字 bigram、追加依存なし）にフォールバック。BM25 は top1 スコアが top2 の `bm25_top1_ratio` 倍以上のときだけ採用
- 参照元ページ: `weave_prompt` で本文と図表内容を織り込んだ文章を生成し `final_output` に格納（1 ページに複数図表が割り当たる場合は 1 回の推論でまとめて織り込み）
- 参照が見つからない図表: 自ページの `final_output` 末尾に内容を追記
- 結果は `output_path` 直下の `<pdf_stem>.jsonl` に保存

共通設定:

- `max_in_flight`: プログラム全体で同時に外部APIへ投げる最大リクエスト数（織り込み推論も同じ pool で並列化）
- 再実行時のskip: 1 パス目は `tmp/<pdf_stem>_tmp.jsonl`、2 パス目は `<pdf_stem>.jsonl` の `book` / `page`
- 織り込みプロンプト等を調整して 2 パス目だけやり直す場合: `<pdf_stem>.jsonl` を削除して再実行（`tmp/` が残っていれば OCR は再実行されない）
- 失敗記録: `<pdf_stem>.failures.jsonl`（`stage: ocr` / `stage: weave`）
- 推論結果の表示: `print_step_outputs: true`
- 表示文字数上限: `print_step_output_max_chars`（`0` で省略なし）
- 参照統合: `reference_window`（既定 5）, `bm25_top1_ratio`（既定 1.5）
- 画像化解像度: `dpi`（既定 200。画像トークン量に直結。品質不足なら 300 へ）
- PDF間並列: `pdf_concurrency`（既定 2。複数PDFを同時処理。`max_in_flight` は全PDF共有で維持）

## YAML の利用方法
`yamls/ocr_settings.yaml` の主要項目:
- 推論接続
  - `openrouter`, `openrouter_api_key`, `openrouter_server_url`, `openrouter_model_name`
  - または `SERVER_URL`, `MODEL_NAME`
- 生成設定: `infer_config.max_tokens`, `temperature`, `top_p`
- 並列と再試行: `batch_size`, `max_retries`, `wait_seconds`
- Async 版: `max_in_flight`, `max_connections`, `max_keepalive_connections`, `connect_timeout`, `read_timeout`, `write_timeout`, `pool_timeout`, `http2`
- OCR async デバッグ: `print_step_outputs`, `print_step_output_max_chars`
- プロンプト: `prompts`
  - `ocr_prompt`（単ページ用 `ocr_single_page.txt`）
  - `objects_prompt`
  - `content_understanding_prompt`
  - `weave_prompt`（2 パス目の織り込み用）
  - （必要に応じて）`figure_understanding_prompt`, `table_understanding_prompt`, `photo_understanding_prompt`
- 出力先: `output_path`
