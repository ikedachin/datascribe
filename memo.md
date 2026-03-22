

# references  
pdf sample  
- 文章用  
人工知能戦略本部　開催状況(https://www8.cao.go.jp/cstp/ai/ai_hq/kaisai.html)  
./test_pdfs/gijigaiyo_20250912.pdf  
./test_pdfs/gijigaiyo_20251219.pdf  

- 表用  
https://www8.cao.go.jp/cstp/ai/yosan_7nendo_hosei.pdf  
https://www8.cao.go.jp/cstp/ai/yosan_8nendo_draft.pdf


- 複合
ＡＩ戦略会議（第13回）・ＡＩ制度研究会（第７回）※合同開催
https://www8.cao.go.jp/cstp/ai/ai_senryaku/13kai/13kai.html
https://www8.cao.go.jp/cstp/ai/ai_senryaku/13kai/shiryou2.pdf

- sanitize example
uv run main_2_sanitization.py -s ./test_files/ -p ./yamls/sanitization_settings_pgx.yaml -e txt -t original_text


create_qaの条件
- jsonl,json
ファイル名そのものが保存ファイル（出力はjsonl）
- txt,md
ファイルの親フォルダ名が保存ファイル名（出力はjsonl）
リトライ実装は未のはず。要確認。



python3 main_1_ocr.py -s ./test_source/pdfs/ -p ./yamls/ocr_settings.yaml 

python3 main_2_sanitization.py -s ./test_output/ocr/ -p ./yamls/sanitization_settings.yaml -t text

python3 main_3_create_qa.py -p ./yamls/create_qa_settings.yaml -s ./test_output/test_sanitization/ -t sanitized_text