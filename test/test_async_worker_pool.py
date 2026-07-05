import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx
from PIL import Image

from pipelines.async_httpx_client import AsyncHTTPXChatClient, load_jsonl
from pipelines.create_qa_model_async_pool import AsyncQAPipeline
from pipelines.ocr_model_async_pool import AsyncOcrPipeline
from pipelines.ocr_reference_integration import assign_references
from pipelines.sanitize_pipeline_async_pool import AsyncSanitizePipeline


def write_prompt(path: Path, name: str, text: str) -> str:
    prompt_path = path / name
    prompt_path.write_text(text, encoding="utf-8")
    return str(prompt_path)


def write_png(path: Path) -> Path:
    Image.new("RGB", (12, 12), color="white").save(path)
    return path


class AsyncWorkerPoolTests(unittest.IsolatedAsyncioTestCase):
    async def test_httpx_client_caps_concurrent_requests_and_closes(self):
        active = 0
        max_seen = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal active, max_seen
            active += 1
            max_seen = max(max_seen, active)
            await asyncio.sleep(0.02)
            active -= 1
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
            )

        settings = {
            "SERVER_URL": "http://example.test/v1",
            "MODEL_NAME": "fake",
            "max_in_flight": 2,
            "max_retries": 1,
        }
        async with AsyncHTTPXChatClient(settings, transport=httpx.MockTransport(handler)) as client:
            results = await asyncio.gather(*(client.chat_completion(f"p{i}") for i in range(5)))
            self.assertEqual(results, ["ok"] * 5)
            self.assertLessEqual(max_seen, 2)
        self.assertTrue(client.is_closed)

    async def test_sanitization_continues_after_one_failure_and_preserves_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = {
                "SERVER_URL": "http://example.test/v1",
                "MODEL_NAME": "fake-model",
                "output_path": str(root / "out"),
                "batch_size": 8,
                "max_in_flight": 3,
                "prompts": [
                    {"jp_en_prompt": write_prompt(root, "jp_en.txt", "jp_en {text}")},
                    {"en_jp_prompt": write_prompt(root, "en_jp.txt", "en_jp {text}")},
                    {"eval_prompt": write_prompt(root, "eval.txt", "eval {source_text} {sanitized_text}")},
                ],
            }

            async def handler(request: httpx.Request) -> httpx.Response:
                payload = json.loads(request.content.decode("utf-8"))
                prompt = payload["messages"][0]["content"][0]["text"]
                if "bad" in prompt:
                    return httpx.Response(500, json={"error": "boom"})
                return httpx.Response(200, json={"choices": [{"message": {"content": f"res:{prompt}"}}]})

            rows = [
                {"book": "book.pdf", "page": 1, "original_text": "good"},
                {"book": "book.pdf", "page": 2, "original_text": "bad"},
                {"book": "book.pdf", "page": 3, "original_text": "also good"},
            ]
            client = AsyncHTTPXChatClient(settings, transport=httpx.MockTransport(handler))
            async with AsyncSanitizePipeline(settings, client=client) as pipeline:
                saved = await pipeline.run(rows)
            await client.aclose()

            self.assertEqual(len(saved), 2)
            output_rows = load_jsonl(root / "out" / "sanitized_book.jsonl")
            self.assertEqual(len(output_rows), 2)
            self.assertEqual([row["page"] for row in output_rows], [1, 3])  # 最終出力はpage順
            self.assertIn("sanitized_original_text", output_rows[0])
            self.assertIn("similarity_original_text", output_rows[0])
            self.assertEqual(output_rows[0]["generator"], "fake-model")
            tmp_rows = load_jsonl(root / "out" / "tmp" / "sanitized_book_tmp.jsonl")
            self.assertEqual(len(tmp_rows), 2)  # 中間(tmp)はresume台帳
            failures = load_jsonl(root / "out" / "tmp" / "sanitized_book.failures.jsonl")
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["page"], 2)

    async def test_ocr_worker_pool_runs_steps_saves_schema_and_skips_processed_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = write_png(root / "book_page1.png")
            calls = []
            settings = {
                "SERVER_URL": "http://example.test/v1",
                "MODEL_NAME": "fake-model",
                "output_path": str(root / "out"),
                "max_in_flight": 1,
                "max_retries": 1,
                "print_step_outputs": False,
            }
            prompt_dict = {
                "ocr_prompt": "ocr step",
                "objects_prompt": "objects step",
                "content_understanding_prompt": "content step {ocr}",
            }
            image_counts = []

            async def handler(request: httpx.Request) -> httpx.Response:
                payload = json.loads(request.content.decode("utf-8"))
                prompt = payload["messages"][0]["content"][0]["text"]
                calls.append(prompt)
                image_counts.append(
                    sum(1 for part in payload["messages"][0]["content"] if part.get("type") == "image_url")
                )
                if prompt == "ocr step":
                    content = "ocr-ok"
                elif prompt == "objects step":
                    content = '[["figure", 0, 0, 1000, 1000]]'
                elif prompt == "content step ocr-ok":
                    content = "object-ok"
                else:
                    content = "unexpected"
                return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

            client = AsyncHTTPXChatClient(settings, transport=httpx.MockTransport(handler))
            async with AsyncOcrPipeline(settings, {"MODEL_NAME": "fake-model"}, client=client) as pipeline:
                saved = await pipeline.run_pdf(
                    file_name="book",
                    book_name="book.pdf",
                    image_files=[image_path],
                    prompt_dict=prompt_dict,
                )
                skipped = await pipeline.run_pdf(
                    file_name="book",
                    book_name="book.pdf",
                    image_files=[image_path],
                    prompt_dict=prompt_dict,
                )
            await client.aclose()

            self.assertEqual(calls, ["ocr step", "objects step", "content step ocr-ok"])
            self.assertEqual(image_counts[0], 1)  # 単ページ送り（ペア廃止）
            self.assertEqual(len(saved), 1)
            self.assertEqual(skipped, [])
            output_rows = load_jsonl(root / "out" / "tmp" / "book_tmp.jsonl")
            self.assertEqual(len(output_rows), 1)
            self.assertEqual(output_rows[0]["page"], 1)
            self.assertEqual(output_rows[0]["book"], "book.pdf")
            self.assertEqual(output_rows[0]["ocr_generator"], "fake-model")
            self.assertEqual(output_rows[0]["text"], "ocr-ok")
            self.assertEqual(output_rows[0]["objects_bbox"], '[["figure", 0, 0, 1000, 1000]]')
            self.assertEqual(output_rows[0]["object_content"], ["object-ok"])
            self.assertEqual(output_rows[0]["final_output"], "ocr-ok")
            self.assertNotIn("duplicate_check", output_rows[0])

    async def test_ocr_worker_pool_records_failure_and_continues_other_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_1 = write_png(root / "book_page1.png")
            image_2 = write_png(root / "book_page2.png")
            objects_calls = 0
            settings = {
                "SERVER_URL": "http://example.test/v1",
                "MODEL_NAME": "fake-model",
                "output_path": str(root / "out"),
                "max_in_flight": 1,
                "max_retries": 1,
                "print_step_outputs": False,
            }
            prompt_dict = {
                "ocr_prompt": "ocr step",
                "objects_prompt": "objects step",
            }

            async def handler(request: httpx.Request) -> httpx.Response:
                nonlocal objects_calls
                payload = json.loads(request.content.decode("utf-8"))
                prompt = payload["messages"][0]["content"][0]["text"]
                if prompt == "ocr step":
                    content = "ocr-ok"
                elif prompt == "objects step":
                    objects_calls += 1
                    if objects_calls == 2:
                        return httpx.Response(500, json={"error": "boom"})
                    content = "[]"
                else:
                    content = "unexpected"
                return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

            client = AsyncHTTPXChatClient(settings, transport=httpx.MockTransport(handler))
            async with AsyncOcrPipeline(settings, {"MODEL_NAME": "fake-model"}, client=client) as pipeline:
                saved = await pipeline.run_pdf(
                    file_name="book",
                    book_name="book.pdf",
                    image_files=[image_1, image_2],
                    prompt_dict=prompt_dict,
                )
            await client.aclose()

            self.assertEqual(len(saved), 1)
            output_rows = load_jsonl(root / "out" / "tmp" / "book_tmp.jsonl")
            self.assertEqual(len(output_rows), 1)
            self.assertEqual(output_rows[0]["page"], 1)
            failures = load_jsonl(root / "out" / "tmp" / "book.failures.jsonl")
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["stage"], "ocr")
            self.assertEqual(failures[0]["failed_step"], "objects_bbox")
            self.assertEqual(failures[0]["book"], "book.pdf")
            self.assertEqual(failures[0]["page"], 2)
            self.assertEqual(failures[0]["source"], str(image_2))

    def test_reference_assignment_caption_bm25_and_fallback(self):
        rows = [
            {"page": 1, "text": "図1に示すように心臓は四つの部屋から構成されている。", "object_content": []},
            {
                "page": 2,
                "text": "本文ページ2。",
                "object_content": ["図1 心臓の構造。左心房、左心室、右心房、右心室を示す模式図。"],
            },
            {
                "page": 3,
                "text": "本文ページ3。",
                "object_content": ["腎臓のネフロン構造と糸球体濾過の仕組みを表した模式図。"],
            },
            {"page": 4, "text": "腎臓のネフロン構造では糸球体濾過が行われ、原尿が生成される。", "object_content": []},
            {"page": 5, "text": "The quick brown fox jumps over the lazy dog.", "object_content": ["ξψζωθ"]},
        ]
        assignments = assign_references(rows, window=5)
        by_source = {(a.source_page, a.object_index): a for a in assignments}

        caption_hit = by_source[(2, 0)]
        self.assertEqual(caption_hit.target_page, 1)
        self.assertEqual(caption_hit.method, "caption")

        bm25_hit = by_source[(3, 0)]
        self.assertEqual(bm25_hit.target_page, 4)
        self.assertEqual(bm25_hit.method, "bm25")

        fallback = by_source[(5, 0)]
        self.assertIsNone(fallback.target_page)
        self.assertEqual(fallback.method, "none")

    async def test_ocr_integration_weaves_referenced_objects_into_final_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = {
                "SERVER_URL": "http://example.test/v1",
                "MODEL_NAME": "fake-model",
                "output_path": str(root / "out"),
                "max_in_flight": 2,
                "max_retries": 1,
                "print_step_outputs": False,
                "reference_window": 5,
            }
            prompt_dict = {"weave_prompt": "weave\n{ocr}\n{objects}"}
            rows = [
                {
                    "page": 1,
                    "book": "book.pdf",
                    "text": "自ページ追記対象。",
                    "object_content": ["ξψζωθ"],
                },
                {
                    "page": 2,
                    "book": "book.pdf",
                    "text": "図1に示すように心臓は四つの部屋から構成されている。",
                    "object_content": [],
                },
                {
                    "page": 3,
                    "book": "book.pdf",
                    "text": "本文ページ3。",
                    "object_content": ["図1 心臓の構造。左心房、左心室、右心房、右心室を示す模式図。"],
                },
            ]
            weave_calls = []

            async def handler(request: httpx.Request) -> httpx.Response:
                payload = json.loads(request.content.decode("utf-8"))
                prompt = payload["messages"][0]["content"][0]["text"]
                weave_calls.append(prompt)
                return httpx.Response(200, json={"choices": [{"message": {"content": "WOVEN"}}]})

            client = AsyncHTTPXChatClient(settings, transport=httpx.MockTransport(handler))
            async with AsyncOcrPipeline(settings, {"MODEL_NAME": "fake-model"}, client=client) as pipeline:
                saved = await pipeline.integrate_pdf(
                    file_name="book", book_name="book.pdf", prompt_dict=prompt_dict, rows=rows
                )
                skipped = await pipeline.integrate_pdf(
                    file_name="book", book_name="book.pdf", prompt_dict=prompt_dict, rows=rows
                )
            await client.aclose()

            self.assertEqual(len(saved), 3)
            self.assertEqual(skipped, [])  # 再実行はfinal jsonlのbook/pageでskip
            self.assertEqual(len(weave_calls), 1)
            self.assertIn("図1 心臓の構造", weave_calls[0])

            final_rows = {row["page"]: row for row in load_jsonl(root / "out" / "book.jsonl")}
            self.assertEqual(len(final_rows), 3)
            # 参照元ページ(2)はLLMで織り込んだ文章に置き換わる
            self.assertEqual(final_rows[2]["final_output"], "WOVEN")
            self.assertEqual(final_rows[2]["woven_objects"][0]["source_page"], 3)
            self.assertEqual(final_rows[2]["woven_objects"][0]["method"], "caption")
            # 参照なし図表は自ページのfinal_outputに追記
            self.assertIn("自ページ追記対象。", final_rows[1]["final_output"])
            self.assertIn("ξψζωθ", final_rows[1]["final_output"])
            # 図の載っているページ(3)は本文のまま
            self.assertEqual(final_rows[3]["final_output"], "本文ページ3。")

    async def test_qa_worker_pool_saves_fast_item_before_slow_item_and_skips_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = {
                "SERVER_URL": "http://example.test/v1",
                "MODEL_NAME": "fake-model",
                "output_path": str(root / "out"),
                "max_in_flight": 2,
                "max_retries": 1,
                "prompts": [
                    {"question_prompt": write_prompt(root, "question.txt", "question {text} {random_token}")},
                    {"answer_prompt": write_prompt(root, "answer.txt", "answer {question} {text}")},
                ],
            }

            async def handler(request: httpx.Request) -> httpx.Response:
                payload = json.loads(request.content.decode("utf-8"))
                prompt = payload["messages"][0]["content"][0]["text"]
                if "slow" in prompt:
                    await asyncio.sleep(0.04)
                if prompt.startswith("question"):
                    content = f"<question>{prompt.split()[1]}</question>"
                else:
                    content = f"<answer>{prompt.split()[-1]}</answer>"
                return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

            out = root / "out" / "book.jsonl"
            jobs = [
                {
                    "text": "slow",
                    "source": "slow",
                    "source_files": ["slow.txt"],
                    "source_key": "text:slow",
                    "output_jsonl": out,
                    "status_jsonl": out.with_suffix(".status.jsonl"),
                    "failure_jsonl": out.with_suffix(".failures.jsonl"),
                },
                {
                    "text": "fast",
                    "source": "fast",
                    "source_files": ["fast.txt"],
                    "source_key": "text:fast",
                    "output_jsonl": out,
                    "status_jsonl": out.with_suffix(".status.jsonl"),
                    "failure_jsonl": out.with_suffix(".failures.jsonl"),
                },
            ]

            client = AsyncHTTPXChatClient(settings, transport=httpx.MockTransport(handler))
            async with AsyncQAPipeline(settings, client=client) as pipeline:
                await pipeline.run_jobs(jobs)
                await pipeline.run_jobs(jobs)
            await client.aclose()

            rows = load_jsonl(out)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["question"], "fast")
            self.assertIn("id", rows[0])
            self.assertEqual(rows[0]["qa_generator"], "fake-model")
            statuses = load_jsonl(out.with_suffix(".status.jsonl"))
            self.assertEqual({row["source_key"] for row in statuses}, {"text:slow", "text:fast"})


if __name__ == "__main__":
    unittest.main()
