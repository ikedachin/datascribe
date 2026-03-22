import json
import random
import httpx
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor


class QAPipeline:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.inference_config = dict(settings.get("infer_config", {}))

        if settings.get("openrouter", False):
            api_key = settings.get("openrouter_api_key", "dummy")
            server_url = settings.get("openrouter_server_url", "https://openrouter.ai/api/v1")
            model_name = settings.get("openrouter_model_name", None)
            self.inference_config.update(
                {
                    "API_KEY": api_key,
                    "SERVER_URL": server_url,
                    "MODEL_NAME": model_name,
                }
            )

            self.client = OpenAI(
                base_url=self.inference_config.get("SERVER_URL"),
                api_key=self.inference_config.get("API_KEY"),
            )
            self.runtime_label = "openrouter"
        else:
            api_key = "dummy"
            server_url = settings.get("SERVER_URL", "http://localhost:8000/v1")
            model_name = settings.get("MODEL_NAME", None)
            self.runtime_label = "local"
            self.inference_config.update(
                {
                    "API_KEY": api_key,
                    "SERVER_URL": server_url,
                    "MODEL_NAME": model_name,
                }
            )
            self.client = OpenAI(
                base_url=self.inference_config.get("SERVER_URL"),
                api_key=self.inference_config.get("API_KEY"),
            )

        self.output_dir = (
            Path(settings.get("output_path", "./json_output/qa"))
            .expanduser()
            .resolve()
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompts = self._load_prompts(settings.get("prompts", []))
        self.max_retries = int(settings.get("max_retries", 8))
        self.wait_seconds = float(settings.get("wait_seconds", 1.0))

    def _load_prompts(self, prompts_settings: List[Dict]) -> Dict[str, str]:
        prompts_dict: Dict[str, str] = {}
        for prompt_path_dict in prompts_settings:
            key, prompt_path = list(prompt_path_dict.items())[0]
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompts_dict[key] = f.read()
        return prompts_dict

    def list_files(self, source_path: Union[str, Path, None]) -> Tuple[List[Path], List[Path]]:
        if source_path is None:
            return [], []
        source_path = Path(source_path)
        text_extensions = {".md", ".txt"}
        json_extensions = {".json", ".jsonl"}

        if source_path.is_dir():
            text_files = sorted(
                p for p in source_path.rglob("*") if p.is_file() and p.suffix.lower() in text_extensions
            )
            json_files = sorted(
                p for p in source_path.rglob("*") if p.is_file() and p.suffix.lower() in json_extensions
            )
            return text_files, json_files

        if source_path.is_file():
            suffix = source_path.suffix.lower()
            if suffix in text_extensions:
                return [source_path], []
            if suffix in json_extensions:
                return [], [source_path]
        return [], []

    def build_md_pair(self, index: int, source_files: List[Path]) -> Tuple[str, List[Path]]:
        first_path = source_files[index]
        second_path = source_files[index + 1] if index + 1 < len(source_files) else None
        first_text = first_path.read_text(encoding="utf-8")
        second_text = second_path.read_text(encoding="utf-8") if second_path else ""
        sources = [first_path]
        if second_path is not None:
            sources.append(second_path)
        return f"{first_text}\n\n{second_text}", sources

    def build_json_pair(
        self, index: int, json_data: List[Dict], target_key: Optional[str]
    ) -> Tuple[str, List[int]]:
        first_item = json_data[index]
        second_item = json_data[index + 1] if index + 1 < len(json_data) else None

        first_text = first_item.get(target_key, "")
        second_text = second_item.get(target_key, "") if second_item else ""
        return f"{first_text}\n\n{second_text}", [index, index + 1] if second_item else [index]

    def _infer_text(self, prompt: str) -> str:
        content = [{"type": "text", "text": prompt}]

        max_tokens = int(self.inference_config.get("max_tokens", 2048))
        temperature = self.inference_config.get("temperature", 0)
        top_p = self.inference_config.get("top_p", 1.0)

        last_exc = None
        for i in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.inference_config.get("MODEL_NAME"),
                    messages=[{"role": "user", "content": content}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                try:
                    return response.choices[0].message.content.strip()
                except:
                    return response.choices[0].message.content

            except Exception as e:
                last_exc = e

                # ネットワーク系も軽くリトライ（短め）
                if isinstance(e, (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError)):
                    sleep = min((self.wait_seconds * (2 ** i)) + random.random() * 0.2, 10.0)
                    time.sleep(sleep)
                    continue

                raise  # その他は即死

        # raise RuntimeError(f"Inference retry exhausted. last_error={last_exc}")
        return ""


    def _infer_texts(self, prompts: List[str], batch_size: int) -> List[str]:
        if not prompts:
            return []

        max_workers = max(1, min(batch_size, len(prompts)))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return list(executor.map(self._infer_text, prompts))
            
    def _extract_tag(self, text: str, tag: str) -> str:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"
        if start_tag in text:
            text = text.split(start_tag)[-1]
        if end_tag in text:
            text = text.split(end_tag)[0]
        return text.strip()

    def create_qa(self, text: str) -> Dict[str, str]:
        question_prompt = self.prompts.get("question_prompt", None)
        thinking_prompt = self.prompts.get("thinking_prompt", None)
        answer_prompt = self.prompts.get("answer_prompt", None)
        refine_question_prompt = self.prompts.get("refine_question_prompt", None)
        refine_thinking_prompt = self.prompts.get("refine_thinking_prompt", None)
        refine_answer_prompt = self.prompts.get("refine_answer_prompt", None)

        if not question_prompt or not answer_prompt:
            raise ValueError("question_prompt and answer_prompt must be set in prompts.")

        random_token = "".join(random.sample(text, min(len(text), 10))) if text else ""
        formatted_question_prompt = question_prompt.format(
            text=text, random_token=random_token
        )
        question_text = self._infer_text(formatted_question_prompt)
        question_text = self._extract_tag(question_text, "question")

        if refine_question_prompt:
            formatted_refined_question = refine_question_prompt.format(
                text=text, question=question_text
            )
            refined_question_text = self._infer_text(formatted_refined_question)
            question_text = self._extract_tag(refined_question_text, "question")

        formatted_answer_prompt = answer_prompt.format(text=text, question=question_text)
        answer_text = self._infer_text(formatted_answer_prompt)
        answer_text = self._extract_tag(answer_text, "answer")

        if thinking_prompt:
            formatted_thinking_prompt = thinking_prompt.format(
                text=text, question=question_text, answer=answer_text
            )
            thinking_text = self._infer_text(formatted_thinking_prompt)
            thinking_text = self._extract_tag(thinking_text, "thinking")
        else:
            thinking_text = ""

        if refine_thinking_prompt:
            formatted_refined_thinking = refine_thinking_prompt.format(
                text=text, question=question_text, thought=thinking_text, answer=answer_text
            )
            refined_thinking_text = self._infer_text(formatted_refined_thinking)
            thinking_text = self._extract_tag(refined_thinking_text, "thinking")

        if refine_answer_prompt:
            formatted_refined_answer = refine_answer_prompt.format(
                text=text, question=question_text, thought=thinking_text, answer=answer_text
            )
            refined_answer_text = self._infer_text(formatted_refined_answer)
            answer_text = self._extract_tag(refined_answer_text, "answer")

        return {
            "question": question_text,
            "thinking": thinking_text if thinking_prompt else "",
            "answer": answer_text,
            "refined_thinking": thinking_text if refine_thinking_prompt else "",
            "refined_answer": answer_text if refine_answer_prompt else "",
            "qa_generator": self.inference_config.get("MODEL_NAME", ""),
            # "generator": self.inference_config.get("MODEL_NAME", ""),
        }

    def create_qa_batch(self, texts: List[str], batch_size: int) -> List[Dict[str, str]]:
        if not texts:
            return []

        question_prompt = self.prompts.get("question_prompt", None)
        thinking_prompt = self.prompts.get("thinking_prompt", None)
        answer_prompt = self.prompts.get("answer_prompt", None)
        refine_question_prompt = self.prompts.get("refine_question_prompt", None)
        refine_thinking_prompt = self.prompts.get("refine_thinking_prompt", None)
        refine_answer_prompt = self.prompts.get("refine_answer_prompt", None)

        if not question_prompt or not answer_prompt:
            raise ValueError("question_prompt and answer_prompt must be set in prompts.")

        random_tokens = [
            "".join(random.sample(text, min(len(text), 10))) if text else "" for text in texts
        ]
        question_prompts = [
            question_prompt.format(text=text, random_token=random_token)
            for text, random_token in zip(texts, random_tokens)
        ]
        question_texts = [
            self._extract_tag(text, "question")
            for text in self._infer_texts(question_prompts, batch_size)
        ]

        if refine_question_prompt:
            refine_question_prompts = [
                refine_question_prompt.format(text=text, question=question_text)
                for text, question_text in zip(texts, question_texts)
            ]
            question_texts = [
                self._extract_tag(text, "question")
                for text in self._infer_texts(refine_question_prompts, batch_size)
            ]

        answer_prompts = [
            answer_prompt.format(text=text, question=question_text)
            for text, question_text in zip(texts, question_texts)
        ]
        answer_texts = [
            self._extract_tag(text, "answer")
            for text in self._infer_texts(answer_prompts, batch_size)
        ]

        if thinking_prompt:
            thinking_prompts = [
                thinking_prompt.format(text=text, question=question_text, answer=answer_text)
                for text, question_text, answer_text in zip(texts, question_texts, answer_texts)
            ]
            thinking_texts = [
                self._extract_tag(text, "thinking")
                for text in self._infer_texts(thinking_prompts, batch_size)
            ]
        else:
            thinking_texts = ["" for _ in texts]

        if refine_thinking_prompt:
            refine_thinking_prompts = [
                refine_thinking_prompt.format(
                    text=text, question=question_text, thought=thinking_text, answer=answer_text
                )
                for text, question_text, thinking_text, answer_text in zip(
                    texts, question_texts, thinking_texts, answer_texts
                )
            ]
            thinking_texts = [
                self._extract_tag(text, "thinking")
                for text in self._infer_texts(refine_thinking_prompts, batch_size)
            ]

        if refine_answer_prompt:
            refine_answer_prompts = [
                refine_answer_prompt.format(
                    text=text, question=question_text, thought=thinking_text, answer=answer_text
                )
                for text, question_text, thinking_text, answer_text in zip(
                    texts, question_texts, thinking_texts, answer_texts
                )
            ]
            answer_texts = [
                self._extract_tag(text, "answer")
                for text in self._infer_texts(refine_answer_prompts, batch_size)
            ]

        results = []
        for question_text, thinking_text, answer_text in zip(
            question_texts, thinking_texts, answer_texts
        ):
            if question_text and answer_text:
                results.append(
                    {
                        "question": question_text,
                        "thinking": thinking_text if thinking_prompt else "",
                        "answer": answer_text,
                        "refined_thinking": thinking_text if refine_thinking_prompt else "",
                        "refined_answer": answer_text if refine_answer_prompt else "",
                        "qa_generator": self.inference_config.get("MODEL_NAME", ""),
                        "generator": self.inference_config.get("MODEL_NAME", ""),
                    }
                )
            else:
                pass
        return results

    def _append_jsonl(self, save_path: Path, result: Dict) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "a", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
            f.write("\n")
