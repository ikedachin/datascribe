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


def _extract_page_num(image_path):
    """画像ファイル名からページ番号を抽出する。抽出できない場合はNone。"""
    try:
        return int(float(Path(image_path).stem.split("page")[-1]))
    except Exception:
        return None


def _upsert_page_result(results_by_image, image_path, key, value, generator, book):
    image_key = str(image_path)
    if image_key not in results_by_image:
        results_by_image[image_key] = {
            "page": _extract_page_num(image_path),
            "book": book,
            "ocr_generator": generator,
        }
    results_by_image[image_key][key] = value


def _save_pdf_results(output_path, file_name, results_by_image):
    save_path = Path(output_path) / f"{file_name}.jsonl"
    with open(save_path, "w", encoding="utf-8") as jsonl_file:
        for result in results_by_image.values():
            jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")


def cleanup_output_images(output_dir):
    """output_dir 配下の画像ファイルを削除する。"""
    output_path = Path(output_dir)
    if not output_path.exists() or not output_path.is_dir():
        return

    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    deleted_count = 0

    for file_path in output_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in image_suffixes:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                tqdm.tqdm.write(msg_error(f"Failed to delete image file: {file_path} ({e})"))

    tqdm.tqdm.write(msg_info(f"Cleaned up {deleted_count} image file(s) in {output_path}"))


# ------------------------------------
# Main function
# ------------------------------------
def main(args, output_dir="./imgs", dpi=300):
    # ------------------------------------
    # pdfs settings
    # ------------------------------------
    settings = load_settings(args.settings_path)


    if args.source_pdfs_path is None:
        print(msg_error("PDFs path is not provided. Please pass <PDFs_PATH>."), file=sys.stderr)
        sys.exit(1)
    else:
        settings['pdfs_path'] = args.source_pdfs_path

    # ------------------------------------
    # create output path
    # ------------------------------------
    output_path = settings.get("output_path", "./outputs/json/result.json")

    if not Path(output_path).exists():
        print(msg_info(f"Creating output directory: {Path(output_path)}"))
        Path(output_path).mkdir(parents=True, exist_ok=True)

    # ------------------------------------
    # API settings
    # ------------------------------------
    inference_config = get_inference_config(settings)

    # ------------------------------------
    # Get PDF files
    # ------------------------------------
    pdf_files = list_pdf_files(args.source_pdfs_path)
    print(msg_success(f"Found {len(pdf_files)} PDF file(s) for path: {args.source_pdfs_path}"))
    if not pdf_files:
        print(msg_error(f"No PDF files found for path: {args.source_pdfs_path}"), file=sys.stderr)
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
        book_name = pdf_path.name
        inference_config['file_name'] = file_name
        results_by_image = {}
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
            for img, res in zip(batched_images[:-1], ocr_response):
                _upsert_page_result(
                    results_by_image,
                    img,
                    "text",
                    res,
                    inference_config.get("MODEL_NAME", ""),
                    book_name,
                )

       
            # ------------------------------------
            # Object Detection
            # ------------------------------------
            objects_bbox = pipeline.batched_infer(
                'objects', 
                prompt=prompt_dict.get("objects_prompt"), 
                batched_images=batched_images,
                previous_result=ocr_response
                )
            for img, res in zip(batched_images[:-1], objects_bbox):
                _upsert_page_result(
                    results_by_image,
                    img,
                    "objects_bbox",
                    res,
                    inference_config.get("MODEL_NAME", ""),
                    book_name,
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
            for img, res in zip(batched_images[:-1], object_understanding_result):
                _upsert_page_result(
                    results_by_image,
                    img,
                    "object_content",
                    res,
                    inference_config.get("MODEL_NAME", ""),
                    book_name,
                )

            print(msg_debug(f"Response(object_content): \n{object_understanding_result}..."))


        # ------------------------------------
        # これは最後
        # ------------------------------------
        ocr_response = pipeline.batched_infer(
            'ocr', 
            prompt=prompt_dict.get("duplicate_check_prompt"), 
            batched_images=batched_images,
            previous_result=ocr_response
            )
        for img, res in zip(batched_images[:-1], ocr_response):
            _upsert_page_result(
                results_by_image,
                img,
                "duplicate_check",
                res,
                inference_config.get("MODEL_NAME", ""),
                book_name,
            )

        _save_pdf_results(output_path, file_name, results_by_image)
        tqdm.tqdm.write(msg_success(f"Saved JSONL: {Path(output_path) / f'{file_name}.jsonl'}"))

    cleanup_output_images(output_dir)


if __name__ == "__main__":
    print(msg_info("Hello from vlm ocr!"))
    parser = argparse.ArgumentParser(description="Run OCR on PDF files.")
    parser.add_argument(
        "-s", "--source_pdfs_path",
        nargs="?",
        default="./test_pdfs/aiplan_g_20251223.pdf",
        help="Path to a PDF file or a directory containing PDFs",
    )
    parser.add_argument(
        "-p", "--settings_path",
        type=str,
        default="./yamls/ocr_settings.yaml",
        help="Path to the YAML settings file",
    )
    args = parser.parse_args()


    main(args)
    
    print(msg_info(f"Processing completed"))
