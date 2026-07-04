# OCR

## 画像の処理方法

### 基本的方針
文章のつながりを優先するため、画像は2枚ずつ処理する。一旦重複させ、あとから整理する。
- 1枚目と2枚目
- 2枚目と3枚目
- 3枚目と4枚目

### 処理のパイプライン
1. OCR
二枚ずつOCRする。文章が2枚にまたがっているときは改行させない指示

1. Object Detection
二枚ずつ表、図、などを取得する

1. 表や図の理解
Object Detectionで得た画像から内容を正確に読み取る（2枚ずつ）

1. 文章の再構築
文章の重複削除
表や図の重複削除

1. JSONLに成形

## Async worker pool 版

`main_1_ocr_async_pool.py` では、`pipelines/ocr_model_worker_pool.py` がページ item ごとの worker pool を管理する。各 item は同じ worker が最後まで担当する。

1. OCR
1. Object Detection (`objects_bbox`)
1. 表や図の理解 (`object_content`)
1. 重複チェック (`duplicate_check`)
1. JSONLに成形

出力 JSONL の主なキーは `book`, `page`, `text`, `objects_bbox`, `object_content`, `duplicate_check`, `ocr_generator`。失敗した item は `<pdf_stem>.failures.jsonl` に保存し、再実行時は既存出力の `book` / `page` を見て処理済みをskipする。

各ステップの推論結果は標準出力にも表示する。YAMLで `print_step_outputs: false` にすると非表示、`print_step_output_max_chars: 0` にすると省略なしで表示する。
