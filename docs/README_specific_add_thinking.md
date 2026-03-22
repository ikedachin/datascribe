# specific_add_thinking.py

## 概要
`main_3_create_qa.py` などで作成済みの QA JSON を読み込み、`thinking` と `messages` を追加して JSONL に出力します。

## 実行例
```bash
python specific_add_thinking.py \
  -i ./input_qa_json_dir \
  -o ./output_with_thinking \
  -r ./test_source/mds \
  -p ./your_add_thinking_settings.yaml
```

## 入力
- `-i/--input_path`: `.json` ファイル群を含むディレクトリ
  - 各 JSON は「配列」で、要素は次の形を想定:
```json
[
  {
    "source_file": "aaa_0001.md",
    "questions": [
      {"question": "...", "answer": "..."}
    ]
  }
]
```
- `-r/--reference`: 参照Markdownのルート
  - 実際の参照先は `input_file_stem` を `_qa -> _topics` 置換したサブディレクトリ

## 出力
- `-o/--output_path` 配下に `<入力jsonファイル名>.jsonl`
- 各行の主なキー:
  - `id`, `question`, `answer`, `thinking`, `metadata`, `messages`
- `messages` 形式:
```json
[
  {"role": "user", "content": "<question>"},
  {"role": "assistant", "content": "<think>...</think><answer>"}
]
```
- 再実行制御用に `files_added_thinking.txt` を同ディレクトリへ追記

## YAML 設定
最低限必要なキー例:
```yaml
openrouter_api_key: "..."
openrouter_server_url: "https://openrouter.ai/api/v1"
openrouter_model_name: "qwen/qwen3-30b-a3b-thinking-2507"
batch_size: 4
start_idx: 0
max_retries: 3
wait_seconds: 5
infer_config:
  max_tokens: 4096
  temperature: 0
  top_p: 1.0
prompts:
  create_thinking: ./prompts/add_thinking/create_thinking.txt
  refine_thinking: ./prompts/add_thinking/refine_thinking.txt
```
