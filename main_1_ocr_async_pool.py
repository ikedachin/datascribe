import argparse
import asyncio
import sys
from pathlib import Path

import tqdm

from commons.utils_msg import msg_error, msg_info, msg_success
from commons.utils_pdf import cleanup_output_images, convert_pdf_to_images, list_pdf_files
from commons.utils_read_prompt import get_prompts
from commons.utils_settings import get_inference_config, load_settings
from pipelines.ocr_model_async_pool import AsyncOcrPipeline

DEFAULT_DPI = 200
DEFAULT_PDF_CONCURRENCY = 2


async def async_main(args, output_dir: str = "./imgs") -> None:
    settings = load_settings(args.settings_path)
    if args.source_pdfs_path is None:
        print(msg_error("PDFs path is not provided. Please pass <PDFs_PATH>."), file=sys.stderr)
        sys.exit(1)
    settings["pdfs_path"] = args.source_pdfs_path

    dpi = int(settings.get("dpi", DEFAULT_DPI))
    pdf_concurrency = max(1, int(settings.get("pdf_concurrency", DEFAULT_PDF_CONCURRENCY)))

    output_path = Path(settings.get("output_path", "./outputs/json/result.json"))
    output_path.mkdir(parents=True, exist_ok=True)
    inference_config = get_inference_config(settings)
    prompt_dict = get_prompts(settings.get("prompts", {}))

    pdf_files = list_pdf_files(args.source_pdfs_path)
    print(msg_success(f"Found {len(pdf_files)} PDF file(s) for path: {args.source_pdfs_path}"))
    if not pdf_files:
        print(msg_error(f"No PDF files found for path: {args.source_pdfs_path}"), file=sys.stderr)
        sys.exit(1)

    async with AsyncOcrPipeline(settings, inference_config) as pipeline:
        pdf_semaphore = asyncio.Semaphore(pdf_concurrency)
        progress = tqdm.tqdm(total=len(pdf_files), desc=msg_info("PDFs"))

        async def process_one_pdf(pdf_path: Path) -> None:
            try:
                async with pdf_semaphore:
                    file_name = pdf_path.stem
                    book_name = pdf_path.name
                    tqdm.tqdm.write(msg_info(f"Processing PDF: {pdf_path} (dpi={dpi})"))
                    # PDF -> 画像変換は同期処理なので、イベントループを塞がないよう別スレッドで実行
                    image_files = [
                        Path(p)
                        for p in await asyncio.to_thread(convert_pdf_to_images, pdf_path, output_dir, dpi)
                    ]
                    await pipeline.run_pdf(
                        file_name=file_name,
                        book_name=book_name,
                        image_files=image_files,
                        prompt_dict=prompt_dict,
                    )
                    tqdm.tqdm.write(msg_success(f"Saved JSONL: {pipeline.output_jsonl(file_name)}"))
                    await pipeline.integrate_pdf(
                        file_name=file_name,
                        book_name=book_name,
                        prompt_dict=prompt_dict,
                    )
                    tqdm.tqdm.write(msg_success(f"Saved final JSONL: {pipeline.final_jsonl(file_name)}"))
            except Exception as exc:
                tqdm.tqdm.write(msg_error(f"PDF failed: {pdf_path} ({exc})"))
            finally:
                progress.update(1)

        await asyncio.gather(*(process_one_pdf(pdf_path) for pdf_path in pdf_files))
        progress.close()

    cleanup_output_images(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCR with an asyncio worker pool.")
    parser.add_argument(
        "-s",
        "--source_pdfs_path",
        nargs="?",
        default="./test_pdfs/aiplan_g_20251223.pdf",
        help="Path to a PDF file or a directory containing PDFs",
    )
    parser.add_argument(
        "-p",
        "--settings_path",
        type=str,
        default="./yamls/ocr_settings.yaml",
        help="Path to the YAML settings file",
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    print(msg_info("Hello from async vlm ocr worker pool!"))
    main()
    print(msg_info("Processing completed"))
