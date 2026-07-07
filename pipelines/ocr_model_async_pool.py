import asyncio
import ast
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from pipelines.async_httpx_client import AsyncHTTPXChatClient, append_jsonl, load_jsonl
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
        self.bbox_debug_lock = asyncio.Lock()

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

    def bbox_debug_jsonl(self, file_name: str) -> Path:
        """図・表・写真として検出されたbboxを、どのファイル/ページのものか特定できる形で記録するデバッグ用出力。"""
        return self.tmp_dir / f"{file_name}_bbox_debug.jsonl"

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
        *,
        file_name: str | None = None,
        book_name: str | None = None,
        page: Any = None,
    ) -> List[str]:
        parsed = parse_objects_bbox(objects_bbox)
        if not parsed and isinstance(objects_bbox, str) and objects_bbox.strip() not in ("", "[]") and file_name is not None:
            # モデル出力が空/[]以外なのにbboxとして解釈できなかった場合、原因調査用に記録する。
            await self._log_bbox_parse_failure(
                file_name=file_name,
                book_name=book_name,
                page=page,
                raw_objects_bbox=objects_bbox,
            )
        outputs: List[str] = []
        for object_index, bbox in enumerate(parsed):
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 5:
                continue
            cropped, crop_box = crop_image(pair_images[0], bbox)
            if file_name is not None:
                await self._log_bbox_debug(
                    file_name=file_name,
                    book_name=book_name,
                    page=page,
                    object_index=object_index,
                    bbox=bbox,
                    crop_box=crop_box,
                    source_image=pair_images[0],
                )
            formatted_prompt = prompt.format(ocr=previous_text)
            outputs.append(await self._infer(formatted_prompt, [cropped]))
        return outputs

    async def _log_bbox_debug(
        self,
        *,
        file_name: str,
        book_name: str | None,
        page: Any,
        object_index: int,
        bbox: Any,
        crop_box: Dict[str, int],
        source_image: Path,
    ) -> None:
        label = bbox[1] if len(bbox) >= 6 else bbox[0]
        row = {
            "file_name": file_name,
            "book": book_name,
            "page": page,
            "object_index": object_index,
            "label": label,
            "bbox": list(bbox),
            "crop_box_px": crop_box,
            "source_image": str(source_image),
        }
        async with self.bbox_debug_lock:
            append_jsonl(self.bbox_debug_jsonl(file_name), row)

    async def _log_bbox_parse_failure(
        self,
        *,
        file_name: str,
        book_name: str | None,
        page: Any,
        raw_objects_bbox: str,
    ) -> None:
        row = {
            "file_name": file_name,
            "book": book_name,
            "page": page,
            "object_index": None,
            "label": None,
            "bbox": None,
            "crop_box_px": None,
            "source_image": None,
            "parse_error": True,
            "raw_objects_bbox": raw_objects_bbox,
        }
        async with self.bbox_debug_lock:
            append_jsonl(self.bbox_debug_jsonl(file_name), row)

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


_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """```json ... ``` のようなMarkdownコードフェンスで囲まれている場合は中身だけを取り出す。"""
    stripped = text.strip()
    match = _CODE_FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped


def parse_objects_bbox(objects_bbox: Any) -> List[Any]:
    if objects_bbox is None:
        return []
    if isinstance(objects_bbox, str):
        candidate = _strip_code_fence(objects_bbox)
        for loader in (json.loads, ast.literal_eval):
            try:
                objects_bbox = loader(candidate)
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


def crop_image(image_path: Path, box_info: List[Any], margin_ratio: float = 0.1) -> tuple[BytesIO, Dict[str, int]]:
    """bboxで示された領域を、縦横それぞれ margin_ratio 分だけ拡大して切り出す。

    box_info の座標は 0-1000 に正規化された値（左上が (0, 0)、右下が (1000, 1000)）。
    拡大はbboxの中心を基準に、幅・高さそれぞれを (1 + margin_ratio) 倍する。
    戻り値は (切り出した画像, 実ピクセル単位の切り出し範囲) のタプル。
    """
    img = Image.open(image_path)
    img_array = np.array(img)
    height, width = img_array.shape[0], img_array.shape[1]
    if len(box_info) >= 6:
        _, _, x1, y1, x2, y2 = box_info
    else:
        _, x1, y1, x2, y2 = box_info
    x1 = float(x1) / 1000.0 * width
    y1 = float(y1) / 1000.0 * height
    x2 = float(x2) / 1000.0 * width
    y2 = float(y2) / 1000.0 * height
    dx = (x2 - x1) * margin_ratio / 2
    dy = (y2 - y1) * margin_ratio / 2
    s_x = max(0, int(round(x1 - dx)))
    s_y = max(0, int(round(y1 - dy)))
    e_x = min(width, int(round(x2 + dx)))
    e_y = min(height, int(round(y2 + dy)))
    cropped = img_array[s_y:e_y, s_x:e_x]
    cropped_img = Image.fromarray(cropped)
    buffered = BytesIO()
    cropped_img.save(buffered, format="PNG")
    buffered.seek(0)
    crop_box = {"x1": s_x, "y1": s_y, "x2": e_x, "y2": e_y}
    return buffered, crop_box
