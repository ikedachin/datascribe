import argparse
import yaml
import sys
import json
from pathlib import Path

import tqdm
from openai import OpenAI

from commons.utils_imgs import encode_image, crop_image
from commons.utils_pdf import convert_pdf_to_images
from commons.utils_msg import msg_info, msg_error, msg_debug, msg_success
from commons.utils_settings import load_settings, get_inference_config
from commons.utils_read_prompt import get_prompts
from pipelines.ocr_model import OcrPipeline
# from src.ocr_model import generate, OcrPipline


def list_pdf_files(pdfs_path):
    if pdfs_path is None:
        return []
    pdfs_path = Path(pdfs_path)
    if pdfs_path.is_dir():
        return sorted(pdfs_path.glob("*.pdf"))
    if pdfs_path.is_file() and pdfs_path.suffix.lower() == ".pdf":
        return [pdfs_path]
    return []


# ------------------------------------
# Main function
# ------------------------------------
def main(args, output_dir="./imgs", dpi=300):
    # ------------------------------------
    # pdfs settings
    # ------------------------------------
    settings = load_settings(args.settings)


    if args.pdfs_path is None:
        print(msg_error("PDFs path is not provided. Please pass <PDFs_PATH>."), file=sys.stderr)
        sys.exit(1)
    else:
        settings['pdfs_path'] = args.pdfs_path

    # ------------------------------------
    # create output path
    # ------------------------------------
    json_output_path = settings.get("json_output", "./outputs/json/result.json")

    if not Path(json_output_path).exists():
        print(msg_info(f"Creating output directory: {Path(json_output_path).parent}"))
        Path(json_output_path).mkdir(parents=True, exist_ok=True)

    # ------------------------------------
    # API settings
    # ------------------------------------
    inference_config = get_inference_config(settings)

    # ------------------------------------
    # Get PDF files
    # ------------------------------------
    pdf_files = list_pdf_files(args.pdfs_path)
    print(msg_success(f"Found {len(pdf_files)} PDF file(s) for path: {args.pdfs_path}"))
    if not pdf_files:
        print(msg_error(f"No PDF files found for path: {args.pdfs_path}"), file=sys.stderr)
        sys.exit(1)


    # ------------------------------------
    # prompts
    # ------------------------------------ 
    prompts_settings = settings.get("prompts", {})
    print(msg_debug(f"Prompts Settings: {prompts_settings}"))

    prompt_dict = get_prompts(prompts_settings)
    print(msg_success(f"Loaded Prompts: {list(prompt_dict.keys())}"))

    # PDF->IMG
    for pdf_path in tqdm.tqdm(pdf_files, desc=msg_info("Preprocess...")):
        # 推論情報にfile_nameを設定
        file_name = pdf_path.stem
        inference_config['file_name'] = file_name
        tqdm.tqdm.write(msg_info(f"Processing PDF: {pdf_path}"))

        img_files = convert_pdf_to_images(pdf_path, output_dir, dpi=dpi)
        # tqdm.tqdm.write(msg_debug(f"Image files: {img_files}"))

        # ------------------------------------
        # Pipeline初期化
        # ------------------------------------
        pipeline = OcrPipeline(**inference_config)
        print(msg_debug(OcrPipeline.keys))

        for i in tqdm.tqdm(range(0, len(img_files), settings.get("batch_size", 1)), desc=msg_info("Processing images")):
            print()
            print(msg_debug("-" * 60))
            print(msg_debug(f"Processing image: {i + 1} ~ {min(i + settings.get('batch_size', 1), len(img_files))} / {len(img_files)}"))
            print(msg_debug("-" * 60))

            batched_images = img_files[i:min(i + settings.get("batch_size", 1) + 1, len(img_files))]

            if not prompt_dict.get("ocr_prompt"):
                raise ValueError(msg_error("OCR prompt is not defined. {e}"))

            # ------------------------------------
            # OCR処理
            # ------------------------------------
            ocr_response = pipeline.batched_infer('ocr', prompt=prompt_dict.get("ocr_prompt"), batched_images=batched_images)
            for res in ocr_response:
                print(msg_debug(f"Response(ocr): \n{res[:30]}..."))

       
            # ------------------------------------
            # Object Detection
            # ------------------------------------
            objects_bbox = pipeline.batched_infer(
                'objects', 
                prompt=prompt_dict.get("objects_prompt"), 
                batched_images=batched_images,
                previous_result=ocr_response
                )
            # print(msg_debug(f"Response(objects): {objects_bbox=}"))
            # for res in objects_bbox:
            #     print(msg_debug(f"Response(objects): \n{res}..."))

            # ------------------------------------
            # Contents Understanding処理
            # ------------------------------------
            object_understanding_result = pipeline.object_understanding(
                key='object_content',
                prompt=prompt_dict.get("content_understanding_prompt"),
                batched_images=batched_images,
                objects_bbox=objects_bbox,
                previous_results=ocr_response
            )

            print(msg_debug(f"Response(object_content): \n{object_understanding_result}..."))

            if i > 4:
                break

            sys.exit(1)

        # ------------------------------------
        # これは最後
        # ------------------------------------
        ocr_response = pipeline.batched_infer(
            'ocr', 
            prompt=prompt_dict.get("duplicate_check_prompt"), 
            batched_images=batched_images,
            previous_result=ocr_response
            )
#                 result = pipeline.ocr(
#                     prompt=prompts_dict.get("ocr_prompt", ""),
#                     images_path=[img_files[i], img_files[i+1]] if i < len(img_files) - 1 else [img_files[i]],
#                     file_path=Path(pdf_path),
#                     page=i+1,
#                 )

#                 result = pipeline.content_understanding(
#                     prompt=prompts_dict.get("contents_prompt", ""),
#                     images_path=[img_files[i], img_files[i+1]] if i < len(img_files) - 1 else [img_files[i]],
#                     file_path=Path(pdf_path),
#                 )

#                 # ------------------------------------
#                 # Objects Bbox処理
#                 # ------------------------------------
#                 result = pipeline.objects_detection(
#                     prompt=prompts_dict.get("objects_prompt", ""),
#                     images_path=[img_files[i], img_files[i+1]] if i < len(img_files) - 1 else [img_files[i]],
#                     file_path=Path(pdf_path),
#                 )

#                 # ------------------------------------
#                 # figure 処理
#                 # ------------------------------------
#                 result = pipeline.object_understanding(
#                     prompts_dict=prompts_dict,
#                     images_path=[img_files[i], img_files[i+1]] if i < len(img_files) - 1 else [img_files[i]],
#                     file_path=Path(pdf_path),
#                 )

#                 print(result)
#                 print(pipeline.result)

#                 # ------------------------------------
#                 # json情報の初期化
#                 # ------------------------------------
#                 result = pipeline.initialize_result()

#             # ------------------------------------
#             # 重複削除処理
#             # ------------------------------------
#             print("-" * 60)
#             print(f"Processing duplicate removal for file: {Path(pdf_path).stem}{Path(pdf_path).suffix}")
#             result = pipeline.remove_duplicates(
#                 prompt=prompts_dict.get("duplicate_check_prompt", ""),
#             )

#             # ------------------------------------
#             # test出力
#             # ------------------------------------
#             print("-" * 60)
#             print(f"[INFO] JSON出力: \n{pipeline.errors}")

#             if pipeline.errors:
#                 print("[WARN] 処理中にエラーが発生しました。詳細は以下の通りです。")
#                 for error in pipeline.errors:
#                     print(f"       - {error}")
#                 print("-" * 60)

#                 print(f"[INFO] エラーログを {Path(inference_config['output_path'])}/errors_{Path(pdf_path).stem}.json に保存します。")
#                 with open(f"{Path(inference_config['output_path'])}/errors_{Path(pdf_path).stem}.json", "w", encoding="utf-8") as f:
#                     json.dump(pipeline.errors, f, ensure_ascii=False)

# def batched_main(pdf_files, output_dir="./imgs", dpi=300, batch_size=4):
#     # PDF->IMG
#     for pdf_path in tqdm.tqdm(pdf_files, desc="Processing PDFs"):
#         try:
#             img_files = convert_pdf_to_images(pdf_path, output_dir, dpi=dpi)
#             print("-" * 60)
#             print(f"Image files: {img_files}")

#             # ------------------------------------
#             # Pipeline初期化
#             # ------------------------------------
#             pipeline = OcrPipline(**inference_config)

#             for start in tqdm.tqdm(range(0, len(img_files), batch_size), desc="Processing images (batched)"):
#                 print("-" * 60)
#                 print(f"\nProcessing images: {start + 1}-{min(start + batch_size, len(img_files))} / {len(img_files)}")
#                 print("-" * 60)

#                 batch = []
#                 for i in range(start, min(start + batch_size, len(img_files))):
#                     batch.append([img_files[i], img_files[i+1]] if i < len(img_files) - 1 else [img_files[i]])

#                 # ------------------------------------
#                 # OCR処理（バッチ）
#                 # ------------------------------------
#                 results = pipeline.ocr_batch(
#                     prompt=prompts_dict.get("ocr_prompt", ""),
#                     images_paths=batch,
#                     file_path=Path(pdf_path),
#                     start_page=start + 1,
#                 )

#                 print(results)

#             # ------------------------------------
#             # 重複削除処理
#             # ------------------------------------
#             print("-" * 60)
#             print(f"Processing duplicate removal for file: {Path(pdf_path).stem}{Path(pdf_path).suffix}")
#             result = pipeline.remove_duplicates(
#                 prompt=prompts_dict.get("duplicate_check_prompt", ""),
#             )

#             # ------------------------------------
#             # test出力
#             # ------------------------------------
#             print("-" * 60)
#             print(f"[INFO] JSON出力: \n{pipeline.errors}")

#             if pipeline.errors:
#                 print("[WARN] 処理中にエラーが発生しました。詳細は以下の通りです。")
#                 for error in pipeline.errors:
#                     print(f"       - {error}")
#                 print("-" * 60)

#                 print(f"[INFO] エラーログを {Path(inference_config['output_path'])}/errors_{Path(pdf_path).stem}.json に保存します。")
#                 with open(f"{Path(inference_config['output_path'])}/errors_{Path(pdf_path).stem}.json", "w", encoding="utf-8") as f:
#                     json.dump(pipeline.errors, f, ensure_ascii=False)

#         except Exception as e:
#             print(f"[ERROR] エラー: {pdf_path}", file=sys.stderr)
#             print(f"        {e}", file=sys.stderr)
#             img_files = []



if __name__ == "__main__":
    print(msg_info("Hello from ocr-qwen3-vl!"))
    parser = argparse.ArgumentParser(description="Run OCR on PDF files.")
    parser.add_argument(
        "-p", "--pdfs_path",
        nargs="?",
        default="./test_pdfs/aiplan_g_20251223.pdf",
        help="Path to a PDF file or a directory containing PDFs",
    )
    parser.add_argument(
        "-s", "--settings",
        type=str,
        default="./yamls/ocr_settings.yaml",
        help="Path to the YAML settings file",
    )
    args = parser.parse_args()


    main(args)
    # バッチ処理
    # batched_main(pdf_files=pdf_files, batch_size=8)
    
    print(msg_info(f"Processing completed"))
