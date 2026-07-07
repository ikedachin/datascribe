import asyncio
import base64
import random
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import httpx


def get_async_runtime_config(settings: Dict[str, Any]) -> Dict[str, Any]:
    infer_config = dict(settings.get("infer_config", {}))
    if settings.get("openrouter", False):
        api_key = settings.get("openrouter_api_key", "dummy")
        server_url = settings.get("openrouter_server_url", "https://openrouter.ai/api/v1")
        model_name = settings.get("openrouter_model_name")
    else:
        api_key = "dummy"
        server_url = settings.get("SERVER_URL", "http://localhost:8000/v1")
        model_name = settings.get("MODEL_NAME")

    max_in_flight = int(settings.get("max_in_flight") or settings.get("batch_size") or 1)
    max_in_flight = max(1, max_in_flight)

    return {
        "api_key": api_key,
        "server_url": str(server_url).rstrip("/"),
        "model": model_name,
        "max_tokens": int(infer_config.get("max_tokens", settings.get("max_tokens", 2048))),
        "temperature": infer_config.get("temperature", settings.get("temperature", 0)),
        "top_p": infer_config.get("top_p", settings.get("top_p", 1.0)),
        "max_retries": int(settings.get("max_retries", infer_config.get("max_retries", 3))),
        "wait_seconds": float(settings.get("wait_seconds", infer_config.get("wait_seconds", 1.0))),
        "max_in_flight": max_in_flight,
        "max_connections": int(settings.get("max_connections", max_in_flight)),
        "max_keepalive_connections": int(settings.get("max_keepalive_connections", max_in_flight)),
        "connect_timeout": float(settings.get("connect_timeout", 10.0)),
        "read_timeout": float(settings.get("read_timeout", settings.get("timeout", 120.0))),
        "write_timeout": float(settings.get("write_timeout", 120.0)),
        "pool_timeout": float(settings.get("pool_timeout", 30.0)),
        "http2": bool(settings.get("http2", False)),
    }


def encode_image_source(image: Any) -> str:
    if isinstance(image, (str, Path)):
        with open(image, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    if isinstance(image, (bytes, bytearray)):
        return base64.b64encode(bytes(image)).decode("utf-8")
    if isinstance(image, BytesIO):
        return base64.b64encode(image.getvalue()).decode("utf-8")

    try:
        from PIL import Image as PILImage

        if isinstance(image, PILImage.Image):
            buf = BytesIO()
            image.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        pass

    raise TypeError(f"Unsupported image source: {type(image)!r}")


def build_openai_content(
    prompt: str,
    images: Optional[Iterable[Any]] = None,
    previous_text: Optional[str] = None,
) -> list[dict]:
    content = [{"type": "text", "text": prompt}]
    if previous_text:
        content.append({"type": "text", "text": previous_text})

    if images:
        for image in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encode_image_source(image)}"},
                }
            )
    return content


class AsyncHTTPXChatClient:
    def __init__(
        self,
        settings: Dict[str, Any],
        transport: httpx.AsyncBaseTransport | httpx.BaseTransport | None = None,
    ):
        self.config = get_async_runtime_config(settings)
        self.request_semaphore = asyncio.Semaphore(self.config["max_in_flight"])
        timeout = httpx.Timeout(
            connect=self.config["connect_timeout"],
            read=self.config["read_timeout"],
            write=self.config["write_timeout"],
            pool=self.config["pool_timeout"],
        )
        limits = httpx.Limits(
            max_connections=self.config["max_connections"],
            max_keepalive_connections=self.config["max_keepalive_connections"],
        )
        self._client = httpx.AsyncClient(
            timeout=timeout,
            limits=limits,
            http2=self.config["http2"],
            transport=transport,
        )
        self._closed = False

    async def __aenter__(self) -> "AsyncHTTPXChatClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    @property
    def is_closed(self) -> bool:
        return self._closed or self._client.is_closed

    async def aclose(self) -> None:
        if not self._client.is_closed:
            await self._client.aclose()
        self._closed = True

    async def chat_completion(
        self,
        prompt: str,
        *,
        images: Optional[Iterable[Any]] = None,
        previous_text: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        content = build_openai_content(prompt, images=images, previous_text=previous_text)
        payload: Dict[str, Any] = {
            "model": self.config["model"],
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.config["max_tokens"],
            "temperature": self.config["temperature"],
            "top_p": self.config["top_p"],
        }
        if extra_body:
            payload.update(extra_body)

        endpoint = f"{self.config['server_url']}/chat/completions"
        headers = {"Authorization": f"Bearer {self.config['api_key']}"}
        last_exc: Exception | None = None

        for attempt in range(self.config["max_retries"]):
            try:
                async with self.request_semaphore:
                    response = await self._client.post(endpoint, headers=headers, json=payload)
                    response.raise_for_status()
                data = response.json()
                message = data["choices"][0].get("message", {})
                content_text = message.get("content")
                if content_text is None:
                    content_text = message.get("reasoning") or message.get("reasoning_content") or ""
                return str(content_text).strip()
            except Exception as exc:
                last_exc = exc
                if attempt == self.config["max_retries"] - 1:
                    raise
                sleep_seconds = min(
                    self.config["wait_seconds"] * (2**attempt) + random.random() * 0.2,
                    30.0,
                )
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)

        raise RuntimeError(f"chat completion failed: {last_exc}")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")


def write_jsonl(path: Path, rows: list) -> None:
    """rows で全行を書き直す（並び順を保証したい最終出力用）。"""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            # json.dump(row, f, ensure_ascii=False)
            # f.write("\n")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sort_rows_by_page(rows: list) -> list:
    return sorted(rows, key=lambda row: (row.get("page") is None, row.get("page")))


def load_jsonl(path: Path) -> list[Dict[str, Any]]:
    import json

    rows: list[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows
