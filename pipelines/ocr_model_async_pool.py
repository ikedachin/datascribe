import asyncio
import ast
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from pipelines.async_httpx_client import AsyncHTTPXChatClient, load_jsonl
from pipelines.ocr_model_worker_pool import run_ocr_pdf_worker_pool
from pipelines.ocr_reference_integration import run_reference_integration


class AsyncOcrPipeline:
    def __init__(self, settings: Dict[str, Any], inference_config: Dict[str, Any], client: Any | None = None):
        self.settings = settings
        self.inference_config = inference_config
        self.client = client
        self._owns_client = client is None
        self.output_dir = Path(settings.get("output_path", "./outputs/")).expanduser().resolve()
        self.tmp_dir = self.output_dir / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.max_in_flight = max(1, int(settings.get("max_in_flight") or settings.get("batch_size") or 1))
        self.write_lock = asyncio.Lock()

    async def __aenter__(self) -> "AsyncOcrPipeline":
        if self.client is None:
            self.client = AsyncHTTPXChatClient(self.settings)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._owns_client and self.client is not None:
            await self.client.aclose()

    def output_jsonl(self, file_name: str) -> Path:
        """1パス目の中間出力（resume台帳兼、2パス目の入力）。"""
        return self.tmp_dir / f"{file_name}_tmp.jsonl"

    def final_jsonl(self, file_name: str) -> Path:
        """2パス目の最終出力。output_path直下に置く。"""
        return self.output_dir / f"{file_name}.jsonl"

    def failure_jsonl(self, file_name: str) -> Path:
        return self.tmp_dir / f"{file_name}.failures.jsonl"

    def load_processed_keys(self, file_name: str) -> set[str]:
        processed: set[str] = set()
        for row in load_jsonl(self.output_jsonl(file_name)):
            processed.add(self.item_key(row.get("book"), row.get("page")))
        return processed

    def load_final_processed_keys(self, file_name: str) -> set[str]:
        processed: set[str] = set()
        for row in load_jsonl(self.final_jsonl(file_name)):
            processed.add(self.item_key(row.get("book"), row.get("page")))
        return processed

    def item_key(self, book: Any, page: Any) -> str:
        return f"{book}::{page}"

    def extract_page_num(self, image_path: Path) -> int | None:
        return _extract_page_num(image_path)

    async def infer(self, prompt: str, images: List[Any], previous_text: str | None = None) -> str:
        assert self.client is not None
        return await self.client.chat_completion(prompt, images=images, previous_text=previous_text)

    async def _infer(self, prompt: str, images: List[Any], previous_text: str | None = None) -> str:
        return await self.infer(prompt, images, previous_text=previous_text)

    async def object_understanding(
        self,
        prompt: str,
        pair_images: List[Path],
        objects_bbox: Any,
        previous_text: str,
    ) -> List[str]:
        parsed = parse_objects_bbox(objects_bbox)
        outputs: List[str] = []
        for bbox in parsed:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 5:
                continue
            cropped = crop_image(pair_images[0], bbox)
            formatted_prompt = prompt.format(ocr=previous_text)
            outputs.append(await self._infer(formatted_prompt, [cropped]))
        return outputs

    async def run_pdf(
        self,
        *,
        file_name: str,
        book_name: str,
        image_files: List[Path],
        prompt_dict: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        return await run_ocr_pdf_worker_pool(
            pipeline=self,
            file_name=file_name,
            book_name=book_name,
            image_files=image_files,
            prompt_dict=prompt_dict,
        )

    async def integrate_pdf(
        self,
        *,
        file_name: str,
        book_name: str,
        prompt_dict: Dict[str, str],
        rows: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        """全ページ処理後の 2 パス目: 図表の参照判定と final_output の織り込み。"""
        if rows is None:
            rows = load_jsonl(self.output_jsonl(file_name))
        return await run_reference_integration(
            pipeline=self,
            file_name=file_name,
            book_name=book_name,
            rows=rows,
            prompt_dict=prompt_dict,
        )


def _extract_page_num(image_path: Path) -> int | None:
    try:
        return int(float(Path(image_path).stem.split("page")[-1]))
    except Exception:
        return None


def parse_objects_bbox(objects_bbox: Any) -> List[Any]:
    if objects_bbox is None:
        return []
    if isinstance(objects_bbox, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                objects_bbox = loader(objects_bbox)
                break
            except Exception:
                continue
        else:
            return []
    if not isinstance(objects_bbox, list):
        return []
    parsed: List[Any] = []
    for item in objects_bbox:
        if isinstance(item, str):
            for loader in (json.loads, ast.literal_eval):
                try:
                    item = loader(item)
                    break
                except Exception:
                    continue
        parsed.append(item)
    return parsed


def crop_image(image_path: Path, box_info: List[Any]) -> BytesIO:
    img = Image.open(image_path)
    img_array = np.array(img)
    if len(box_info) >= 6:
        _, _, s_x, s_y, e_x, e_y = box_info
    else:
        _, s_x, s_y, e_x, e_y = box_info
    s_x = int(float(s_x) * img_array.shape[1] * 0.9 * 0.001)
    s_y = int(float(s_y) * img_array.shape[0] * 0.9 * 0.001)
    e_x = int(float(e_x) * img_array.shape[1] * 1.1 * 0.001)
    e_y = int(float(e_y) * img_array.shape[0] * 1.1 * 0.001)
    cropped = img_array[s_y:e_y, s_x:e_x]
    cropped_img = Image.fromarray(cropped)
    buffered = BytesIO()
    cropped_img.save(buffered, format="PNG")
    buffered.seek(0)
    return buffered
