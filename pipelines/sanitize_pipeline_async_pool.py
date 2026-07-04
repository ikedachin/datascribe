import asyncio
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pipelines.async_httpx_client import (
    AsyncHTTPXChatClient,
    append_jsonl,
    load_jsonl,
    sort_rows_by_page,
    write_jsonl,
)
from pipelines.async_worker_pool import run_item_worker_pool


class AsyncSanitizePipeline:
    def __init__(self, settings: Dict[str, Any], client: Any | None = None):
        self.settings = settings
        self.client = client
        self._owns_client = client is None
        self.output_dir = Path(settings.get("output_path") or "./json_output/qa").expanduser().resolve()
        self.tmp_dir = self.output_dir / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.prompts = self._load_prompts(settings.get("prompts", []))
        self.target_key = settings.get("target_key") or "text"
        self.max_in_flight = max(1, int(settings.get("max_in_flight") or settings.get("batch_size") or 1))
        self.write_lock = asyncio.Lock()

    async def __aenter__(self) -> "AsyncSanitizePipeline":
        if self.client is None:
            self.client = AsyncHTTPXChatClient(self.settings)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._owns_client and self.client is not None:
            await self.client.aclose()

    def _load_prompts(self, prompts_settings: List[Dict[str, str]]) -> Dict[str, str]:
        prompts: Dict[str, str] = {}
        for item in prompts_settings:
            key, prompt_path = list(item.items())[0]
            if prompt_path and Path(prompt_path).is_file():
                prompts[key] = Path(prompt_path).read_text(encoding="utf-8")
        return prompts

    def result_path_for(self, row: Dict[str, Any]) -> Path:
        """中間出力（完了順の追記。resume台帳）。"""
        book = str(row.get("book") or "unknown").split(".")[0]
        return self.tmp_dir / f"sanitized_{book}_tmp.jsonl"

    def failure_path_for(self, row: Dict[str, Any]) -> Path:
        book = str(row.get("book") or "unknown").split(".")[0]
        return self.tmp_dir / f"sanitized_{book}.failures.jsonl"

    def load_processed_keys(self) -> set[str]:
        processed: set[str] = set()
        for path in self.tmp_dir.glob("sanitized_*_tmp.jsonl"):
            for row in load_jsonl(path):
                key = self.item_key(row)
                if key:
                    processed.add(key)
        return processed

    def finalize_outputs(self) -> None:
        """tmp の全行を page 順にソートして output_path 直下の最終ファイルへ書き直す。"""
        for tmp_path in sorted(self.tmp_dir.glob("sanitized_*_tmp.jsonl")):
            rows = sort_rows_by_page(load_jsonl(tmp_path))
            final_path = self.output_dir / tmp_path.name.replace("_tmp.jsonl", ".jsonl")
            write_jsonl(final_path, rows)

    def item_key(self, row: Dict[str, Any]) -> str:
        return f"{row.get('book')}::{row.get('page')}"

    async def _infer(self, prompt: str) -> str:
        assert self.client is not None
        extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": not bool(self.settings.get("NOTHINK", False))
            }
        }
        return await self.client.chat_completion(prompt, extra_body=extra_body)

    async def sanitize_one(self, row: Dict[str, Any]) -> Dict[str, Any]:
        target_key = self.settings.get("target_key") or self.target_key
        if target_key not in row:
            target_key = "original_text" if "original_text" in row else "text"

        jp_en_prompt = self.prompts.get("jp_en_prompt")
        en_jp_prompt = self.prompts.get("en_jp_prompt")
        refine_prompt = self.prompts.get("refine_prompt")
        eval_prompt = self.prompts.get("eval_prompt")
        if not jp_en_prompt or not en_jp_prompt:
            raise ValueError("jp_en_prompt and en_jp_prompt must be set.")

        source_text = row[target_key]
        en_text = await self._infer(jp_en_prompt.format(text=source_text))
        sanitized_text = await self._infer(en_jp_prompt.format(text=en_text))
        if refine_prompt:
            sanitized_text = await self._infer(
                refine_prompt.format(source_text=source_text, sanitized_text=sanitized_text)
            )

        result = dict(row)
        if eval_prompt:
            eval_point = await self._infer(
                eval_prompt.format(source_text=source_text, sanitized_text=sanitized_text)
            )
            result[f"eval_{target_key}"] = eval_point

        result[f"sanitized_{target_key}"] = sanitized_text
        result[f"similarity_{target_key}"] = self.tfidf_cosine_similarity(source_text, sanitized_text)
        model = self.settings.get("openrouter_model_name") if self.settings.get("openrouter") else self.settings.get("MODEL_NAME")
        result["generator"] = model or "unknown"
        return result

    async def run(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed = self.load_processed_keys()
        pending = [row for row in rows if self.item_key(row) not in processed]
        saved: List[Dict[str, Any]] = []
        worker_count = min(self.max_in_flight, len(pending)) if pending else 0
        if worker_count == 0:
            # 全件skipでも最終ファイルは tmp から再生成しておく（中断後の復旧用）
            self.finalize_outputs()
            return saved

        async def process_row(row: Dict[str, Any], worker_id: int) -> None:
            del worker_id
            try:
                result = await self.sanitize_one(row)
                async with self.write_lock:
                    append_jsonl(self.result_path_for(result), result)
                    saved.append(result)
            except Exception as exc:
                async with self.write_lock:
                    append_jsonl(
                        self.failure_path_for(row),
                        {
                            "stage": "sanitize",
                            "error": str(exc),
                            "source": row.get("source"),
                            "book": row.get("book"),
                            "page": row.get("page"),
                        },
                    )

        await run_item_worker_pool(
            pending,
            worker_count=worker_count,
            process_item=process_row,
            progress_desc=f"Sanitize items ({len(pending)}/{len(rows)} pending)",
        )
        self.finalize_outputs()
        return saved

    def tfidf_cosine_similarity(
        self,
        text_a: str,
        text_b: str,
        ngram_range: Tuple[int, int] = (1, 2),
        analyzer: str = "char",
    ) -> float:
        text_a = text_a or ""
        text_b = text_b or ""

        def _ngrams(text: str, n: int) -> List[str]:
            if analyzer == "word":
                tokens = text.split()
                return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
            return [text[i : i + n] for i in range(len(text) - n + 1)]

        df: Dict[str, int] = {}
        counts_by_doc: List[Dict[str, int]] = []
        for text in [text_a, text_b]:
            counts: Dict[str, int] = {}
            for n in range(ngram_range[0], ngram_range[1] + 1):
                for gram in _ngrams(text, n):
                    counts[gram] = counts.get(gram, 0) + 1
            counts_by_doc.append(counts)
            for gram in counts:
                df[gram] = df.get(gram, 0) + 1
        if not df:
            return 0.0
        idf = {term: math.log((1.0 + 2) / (1.0 + freq)) + 1.0 for term, freq in df.items()}
        vectors = [{term: freq * idf[term] for term, freq in counts.items()} for counts in counts_by_doc]
        if not vectors[0] or not vectors[1]:
            return 0.0
        dot = sum(vectors[0].get(term, 0.0) * vectors[1].get(term, 0.0) for term in vectors[0])
        norm_a = sum(value * value for value in vectors[0].values()) ** 0.5
        norm_b = sum(value * value for value in vectors[1].values()) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
