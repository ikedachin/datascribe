from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import tqdm

from commons.utils_msg import msg_debug, msg_error, msg_info
from pipelines.async_httpx_client import append_jsonl
from pipelines.async_worker_pool import run_item_worker_pool


OCR_STEP = 1
OBJECTS_STEP = 2
OBJECT_CONTENT_STEP = 3
SAVE_STEP = 4
DONE_STEP = 5

STEP_NAMES = {
    OCR_STEP: "ocr",
    OBJECTS_STEP: "objects_bbox",
    OBJECT_CONTENT_STEP: "object_content",
    SAVE_STEP: "save",
    DONE_STEP: "done",
}


@dataclass
class PipelineJob:
    item_id: str
    file_name: str
    book_name: str
    image_path: Path
    pair_images: List[Path]
    prompt_dict: Dict[str, str]
    step: int
    previous_outputs: Dict[str, Any] = field(default_factory=dict)


def build_ocr_jobs(
    *,
    pipeline: Any,
    file_name: str,
    book_name: str,
    image_files: List[Path],
    prompt_dict: Dict[str, str],
) -> List[PipelineJob]:
    processed = pipeline.load_processed_keys(file_name)
    jobs: List[PipelineJob] = []
    for image_path in image_files:
        page = pipeline.extract_page_num(image_path)
        key = pipeline.item_key(book_name, page)
        if key in processed:
            continue
        jobs.append(
            PipelineJob(
                item_id=key,
                file_name=file_name,
                book_name=book_name,
                image_path=image_path,
                pair_images=[image_path],
                prompt_dict=prompt_dict,
                step=OCR_STEP,
            )
        )
    return jobs


async def run_ocr_pdf_worker_pool(
    *,
    pipeline: Any,
    file_name: str,
    book_name: str,
    image_files: List[Path],
    prompt_dict: Dict[str, str],
) -> List[Dict[str, Any]]:
    jobs = build_ocr_jobs(
        pipeline=pipeline,
        file_name=file_name,
        book_name=book_name,
        image_files=image_files,
        prompt_dict=prompt_dict,
    )
    if not jobs:
        return []

    saved: List[Dict[str, Any]] = []
    worker_count = min(pipeline.max_in_flight, len(jobs))

    async def process_job(job: PipelineJob, worker_id: int) -> None:
        del worker_id
        curr = job
        failed_job = job
        try:
            while curr.step < DONE_STEP:
                failed_job = curr
                curr = await run_one_step(pipeline, curr)
            async with pipeline.write_lock:
                append_jsonl(pipeline.output_jsonl(file_name), curr.previous_outputs)
                saved.append(curr.previous_outputs)
        except Exception as exc:
            print_step_error(pipeline, failed_job, exc)
            async with pipeline.write_lock:
                append_jsonl(
                    pipeline.failure_jsonl(file_name),
                    {
                        "stage": "ocr",
                        "failed_step": step_name(failed_job.step),
                        "error": str(exc),
                        "source": str(failed_job.image_path),
                        "book": book_name,
                        "page": pipeline.extract_page_num(failed_job.image_path),
                    },
                )

    await run_item_worker_pool(
        jobs,
        worker_count=worker_count,
        process_item=process_job,
        progress_desc=msg_info(f"OCR pages [{file_name}]"),
    )
    return saved


async def run_one_step(pipeline: Any, job: PipelineJob) -> PipelineJob:
    outputs = dict(job.previous_outputs)

    if job.step == OCR_STEP:
        ocr_prompt = job.prompt_dict.get("ocr_prompt")
        if not ocr_prompt:
            raise ValueError("ocr_prompt is not defined.")
        text = await pipeline.infer(ocr_prompt, job.pair_images)
        print_step_result(pipeline, job, "text", text)
        outputs.update(
            {
                "page": pipeline.extract_page_num(job.image_path),
                "book": job.book_name,
                "ocr_generator": pipeline.inference_config.get("MODEL_NAME", ""),
                "text": text,
            }
        )
        return replace_step(job, OBJECTS_STEP, outputs)

    if job.step == OBJECTS_STEP:
        objects_prompt = job.prompt_dict.get("objects_prompt")
        if objects_prompt:
            outputs["objects_bbox"] = await pipeline.infer(
                objects_prompt,
                job.pair_images,
                previous_text=outputs["text"],
            )
        else:
            outputs["objects_bbox"] = "[]"
        print_step_result(pipeline, job, "objects_bbox", outputs["objects_bbox"])
        return replace_step(job, OBJECT_CONTENT_STEP, outputs)

    if job.step == OBJECT_CONTENT_STEP:
        content_prompt = job.prompt_dict.get("content_understanding_prompt")
        if content_prompt:
            outputs["object_content"] = await pipeline.object_understanding(
                content_prompt,
                job.pair_images,
                outputs.get("objects_bbox"),
                outputs["text"],
            )
        else:
            outputs["object_content"] = []
        print_step_result(pipeline, job, "object_content", outputs["object_content"])
        return replace_step(job, SAVE_STEP, outputs)

    if job.step == SAVE_STEP:
        outputs["final_output"] = outputs["text"]
        return replace_step(job, DONE_STEP, outputs)

    raise ValueError(f"Unknown OCR pipeline step: {job.step}")


def replace_step(job: PipelineJob, step: int, outputs: Dict[str, Any]) -> PipelineJob:
    return PipelineJob(
        item_id=job.item_id,
        file_name=job.file_name,
        book_name=job.book_name,
        image_path=job.image_path,
        pair_images=job.pair_images,
        prompt_dict=job.prompt_dict,
        step=step,
        previous_outputs=outputs,
    )


def step_name(step: int) -> str:
    return STEP_NAMES.get(step, f"unknown_step_{step}")


def print_step_result(pipeline: Any, job: PipelineJob, output_key: str, value: Any) -> None:
    if not bool(pipeline.settings.get("print_step_outputs", True)):
        return
    page = pipeline.extract_page_num(job.image_path)
    images = ", ".join(str(path) for path in job.pair_images)
    tqdm.tqdm.write(msg_debug(f"OCR step result: book={job.book_name} page={page} step={step_name(job.step)} key={output_key}"))
    tqdm.tqdm.write(msg_debug(f"OCR step images: {images}"))
    tqdm.tqdm.write(format_step_value(value, max_chars=int(pipeline.settings.get("print_step_output_max_chars", 4000))))


def print_step_error(pipeline: Any, job: PipelineJob, exc: Exception) -> None:
    if not bool(pipeline.settings.get("print_step_outputs", True)):
        return
    page = pipeline.extract_page_num(job.image_path)
    tqdm.tqdm.write(
        msg_error(
            f"OCR step failed: book={job.book_name} page={page} "
            f"step={step_name(job.step)} error={exc}"
        )
    )


def format_step_value(value: Any, *, max_chars: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = repr(value)
    if max_chars > 0 and len(text) > max_chars:
        return f"{text[:max_chars]}\n... <truncated {len(text) - max_chars} chars>"
    return text
