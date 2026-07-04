import asyncio
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List

from pipelines.async_httpx_client import AsyncHTTPXChatClient, append_jsonl, load_jsonl
from pipelines.async_worker_pool import run_item_worker_pool


class AsyncQAPipeline:
    def __init__(self, settings: Dict[str, Any], client: Any | None = None):
        self.settings = settings
        self.client = client
        self._owns_client = client is None
        self.output_dir = Path(settings.get("output_path", "./json_output/qa")).expanduser().resolve()
        self.tmp_dir = self.output_dir / "tmp"
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.prompts = self._load_prompts(settings.get("prompts", []))
        self.max_in_flight = max(1, int(settings.get("max_in_flight") or settings.get("batch_size") or 1))
        self.write_lock = asyncio.Lock()

    async def __aenter__(self) -> "AsyncQAPipeline":
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
            prompts[key] = Path(prompt_path).read_text(encoding="utf-8")
        return prompts

    def _extract_tag(self, text: str, tag: str) -> str:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        if start_tag in text:
            text = text.split(start_tag)[-1]
        if end_tag in text:
            text = text.split(end_tag)[0]
        return text.strip()

    async def _infer(self, prompt: str) -> str:
        assert self.client is not None
        return await self.client.chat_completion(prompt)

    async def create_qa_one(self, text: str) -> Dict[str, str]:
        question_prompt = self.prompts.get("question_prompt")
        thinking_prompt = self.prompts.get("thinking_prompt")
        answer_prompt = self.prompts.get("answer_prompt")
        refine_question_prompt = self.prompts.get("refine_question_prompt")
        refine_thinking_prompt = self.prompts.get("refine_thinking_prompt")
        refine_answer_prompt = self.prompts.get("refine_answer_prompt")
        if not question_prompt or not answer_prompt:
            raise ValueError("question_prompt and answer_prompt must be set.")

        random_token = "".join(random.sample(text, min(len(text), 10))) if text else ""
        question_text = self._extract_tag(
            await self._infer(question_prompt.format(text=text, random_token=random_token)),
            "question",
        )

        if refine_question_prompt:
            question_text = self._extract_tag(
                await self._infer(refine_question_prompt.format(text=text, question=question_text)),
                "question",
            )

        answer_text = self._extract_tag(
            await self._infer(answer_prompt.format(text=text, question=question_text)),
            "answer",
        )

        if thinking_prompt:
            thinking_text = self._extract_tag(
                await self._infer(
                    thinking_prompt.format(text=text, question=question_text, answer=answer_text)
                ),
                "thinking",
            )
        else:
            thinking_text = ""

        if refine_thinking_prompt:
            thinking_text = self._extract_tag(
                await self._infer(
                    refine_thinking_prompt.format(
                        text=text,
                        question=question_text,
                        thought=thinking_text,
                        answer=answer_text,
                    )
                ),
                "thinking",
            )

        if refine_answer_prompt:
            answer_text = self._extract_tag(
                await self._infer(
                    refine_answer_prompt.format(
                        text=text,
                        question=question_text,
                        thought=thinking_text,
                        answer=answer_text,
                    )
                ),
                "answer",
            )

        model = self.settings.get("openrouter_model_name") if self.settings.get("openrouter") else self.settings.get("MODEL_NAME")
        return {
            "question": question_text,
            "thinking": thinking_text if thinking_prompt else "",
            "answer": answer_text,
            "refined_thinking": thinking_text if refine_thinking_prompt else "",
            "refined_answer": answer_text if refine_answer_prompt else "",
            "qa_generator": model or "",
            "generator": model or "",
        }

    def load_processed_source_keys(self, output_jsonl: Path, status_jsonl: Path) -> set[str]:
        processed = {row["source_key"] for row in load_jsonl(status_jsonl) if row.get("source_key")}
        for row in load_jsonl(output_jsonl):
            source_files = row.get("source_files")
            if isinstance(source_files, list) and len(source_files) == 1:
                processed.add(f"text:{source_files[0]}")
        return processed

    async def run_jobs(self, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pending: List[Dict[str, Any]] = []
        for job in jobs:
            processed = self.load_processed_source_keys(job["output_jsonl"], job["status_jsonl"])
            if job["source_key"] not in processed:
                pending.append(job)

        worker_count = min(self.max_in_flight, len(pending)) if pending else 0
        saved: List[Dict[str, Any]] = []
        if worker_count == 0:
            return saved

        async def process_job(job: Dict[str, Any], worker_id: int) -> None:
            del worker_id
            try:
                result = await self.create_qa_one(job["text"])
                result["source_files"] = job["source_files"]
                result["id"] = str(uuid.uuid4())
                async with self.write_lock:
                    append_jsonl(job["output_jsonl"], result)
                    append_jsonl(job["status_jsonl"], {"source_key": job["source_key"]})
                    saved.append(result)
            except Exception as exc:
                async with self.write_lock:
                    append_jsonl(
                        job["failure_jsonl"],
                        {
                            "stage": "create_qa",
                            "error": str(exc),
                            "source": job.get("source"),
                            "source_key": job.get("source_key"),
                        },
                    )

        await run_item_worker_pool(
            pending,
            worker_count=worker_count,
            process_item=process_job,
            progress_desc=f"Create Q&A ({len(pending)}/{len(jobs)} pending)",
        )
        return saved
