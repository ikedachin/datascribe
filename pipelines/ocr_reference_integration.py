"""図・表・写真の参照ページ判定と本文への織り込み（OCR パイプラインの 2 パス目）。

1 パス目で保存された中間 JSONL（ページ単位の text / object_content）を入力に、
各図表について「前後 window ページ以内で参照しているページ」を
キャプション番号マッチ -> BM25（文字 bigram）フォールバックの順で判定する。
参照元ページには weave_prompt で本文と図表内容を織り込んだ final_output を生成し、
参照が見つからない図表は自ページの final_output に追記する。
"""

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import tqdm

from commons.utils_msg import msg_debug, msg_error, msg_info
from pipelines.async_httpx_client import append_jsonl, load_jsonl, sort_rows_by_page, write_jsonl
from pipelines.async_worker_pool import run_item_worker_pool


DEFAULT_REFERENCE_WINDOW = 5
DEFAULT_BM25_TOP1_RATIO = 1.5

CAPTION_PATTERN = re.compile(
    r"(?:図|表|写真|グラフ|fig(?:ure)?\.?|table|photo)\s*[0-9]+(?:[-‐‑–.．][0-9]+)*",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    """全角英数字の半角化・小文字化・空白除去を行い、番号マッチを安定させる。"""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"\s+", "", text)
    return text.lower()


def extract_caption_ids(text: str) -> List[str]:
    """「図3」「表2-1」「Figure 4」等のキャプション番号を正規化して抽出する。"""
    normalized = normalize_text(text)
    ids = []
    for match in CAPTION_PATTERN.findall(normalized):
        canonical = re.sub(r"[-‐‑–.．]", "-", match)
        if canonical not in ids:
            ids.append(canonical)
    return ids


def char_bigrams(text: str) -> List[str]:
    normalized = normalize_text(text)
    if len(normalized) < 2:
        return [normalized] if normalized else []
    return [normalized[i : i + 2] for i in range(len(normalized) - 1)]


class BM25:
    """小規模コーパス（前後数ページ）向けの素朴な Okapi BM25。"""

    def __init__(self, documents: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = len(documents)
        self.doc_lens = [len(doc) for doc in documents]
        self.avg_len = (sum(self.doc_lens) / self.doc_count) if self.doc_count else 0.0
        self.term_freqs: List[Dict[str, int]] = []
        doc_freq: Dict[str, int] = {}
        for doc in documents:
            tf: Dict[str, int] = {}
            for term in doc:
                tf[term] = tf.get(term, 0) + 1
            self.term_freqs.append(tf)
            for term in tf:
                doc_freq[term] = doc_freq.get(term, 0) + 1
        self.idf = {
            term: math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def score(self, query_terms: List[str], index: int) -> float:
        if not self.doc_count or self.avg_len <= 0:
            return 0.0
        tf = self.term_freqs[index]
        doc_len = self.doc_lens[index]
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            freq = tf[term]
            denom = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_len)
            score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / denom
        return score


@dataclass
class ObjectAssignment:
    source_page: Any
    object_index: int
    content: str
    target_page: Any = None  # None = 参照ページなし（自ページに追記）
    method: str = "none"  # "caption" / "bm25" / "none"


@dataclass
class WeaveJob:
    item_id: str
    page: Any
    base_text: str
    objects: List[ObjectAssignment] = field(default_factory=list)


def assign_references(
    rows: List[Dict[str, Any]],
    *,
    window: int = DEFAULT_REFERENCE_WINDOW,
    bm25_top1_ratio: float = DEFAULT_BM25_TOP1_RATIO,
) -> List[ObjectAssignment]:
    """各図表の参照元ページを決める。候補は自ページを除く前後 window ページ。"""
    pages = [row for row in rows if row.get("page") is not None]
    pages.sort(key=lambda row: row["page"])
    normalized_texts = {row["page"]: normalize_text(row.get("text", "")) for row in pages}

    assignments: List[ObjectAssignment] = []
    for row in pages:
        contents = row.get("object_content") or []
        for index, content in enumerate(contents):
            if not str(content).strip():
                continue
            assignment = ObjectAssignment(
                source_page=row["page"], object_index=index, content=str(content)
            )
            candidates = [
                other
                for other in pages
                if other["page"] != row["page"]
                and abs(other["page"] - row["page"]) <= window
            ]
            if candidates:
                target = _match_by_caption(assignment, candidates, normalized_texts)
                if target is None:
                    target = _match_by_bm25(assignment, candidates, bm25_top1_ratio)
                if target is not None:
                    assignment.target_page = target
            assignments.append(assignment)
    return assignments


def _match_by_caption(
    assignment: ObjectAssignment,
    candidates: List[Dict[str, Any]],
    normalized_texts: Dict[Any, str],
) -> Any:
    caption_ids = extract_caption_ids(assignment.content)
    if not caption_ids:
        return None
    matched_pages = []
    for candidate in candidates:
        text = normalized_texts.get(candidate["page"], "")
        if any(cid in text for cid in caption_ids):
            matched_pages.append(candidate["page"])
    if not matched_pages:
        return None
    assignment.method = "caption"
    return min(matched_pages, key=lambda page: (abs(page - assignment.source_page), page))


def _match_by_bm25(
    assignment: ObjectAssignment,
    candidates: List[Dict[str, Any]],
    top1_ratio: float,
) -> Any:
    documents = [char_bigrams(candidate.get("text", "")) for candidate in candidates]
    if not any(documents):
        return None
    bm25 = BM25(documents)
    query = char_bigrams(assignment.content)
    scores = sorted(
        ((bm25.score(query, i), candidates[i]["page"]) for i in range(len(candidates))),
        key=lambda pair: pair[0],
        reverse=True,
    )
    top1_score, top1_page = scores[0]
    if top1_score <= 0:
        return None
    if len(scores) > 1 and scores[1][0] > 0 and top1_score < top1_ratio * scores[1][0]:
        return None
    assignment.method = "bm25"
    return top1_page


def build_weave_jobs(
    rows: List[Dict[str, Any]],
    assignments: List[ObjectAssignment],
    *,
    book_name: str,
) -> Tuple[Dict[Any, WeaveJob], Dict[Any, str]]:
    """参照元ページごとの織り込みジョブと、全ページの初期 final_output を作る。

    - 参照が見つからなかった図表は自ページの final_output 末尾に追記する。
    - 参照元ページには「自ページ追記後のテキスト」を土台に weave ジョブを作る。
    """
    final_outputs: Dict[Any, str] = {}
    for row in rows:
        final_outputs[row.get("page")] = str(row.get("text", ""))

    for assignment in assignments:
        if assignment.target_page is not None:
            continue
        base = final_outputs.get(assignment.source_page, "")
        final_outputs[assignment.source_page] = (
            f"{base}\n\n【図・表・写真の内容（page {assignment.source_page}）】\n"
            f"{assignment.content}"
        ).strip()

    weave_jobs: Dict[Any, WeaveJob] = {}
    for assignment in assignments:
        if assignment.target_page is None:
            continue
        job = weave_jobs.get(assignment.target_page)
        if job is None:
            job = WeaveJob(
                item_id=f"{book_name}::{assignment.target_page}::weave",
                page=assignment.target_page,
                base_text=final_outputs.get(assignment.target_page, ""),
            )
            weave_jobs[assignment.target_page] = job
        job.objects.append(assignment)
    return weave_jobs, final_outputs


def format_objects_for_prompt(objects: List[ObjectAssignment]) -> str:
    blocks = []
    for obj in objects:
        blocks.append(f"### page {obj.source_page} の図・表・写真\n{obj.content}")
    return "\n\n".join(blocks)


async def run_reference_integration(
    *,
    pipeline: Any,
    file_name: str,
    book_name: str,
    rows: List[Dict[str, Any]],
    prompt_dict: Dict[str, str],
) -> List[Dict[str, Any]]:
    """2 パス目本体。final JSONL に書き込んだ行のリストを返す。"""
    settings = pipeline.settings
    window = int(settings.get("reference_window", DEFAULT_REFERENCE_WINDOW))
    ratio = float(settings.get("bm25_top1_ratio", DEFAULT_BM25_TOP1_RATIO))

    processed = pipeline.load_final_processed_keys(file_name)
    pending_rows = [
        row for row in rows
        if pipeline.item_key(row.get("book"), row.get("page")) not in processed
    ]
    if not pending_rows:
        return []

    assignments = assign_references(rows, window=window, bm25_top1_ratio=ratio)
    weave_jobs, final_outputs = build_weave_jobs(rows, assignments, book_name=book_name)

    weave_prompt = prompt_dict.get("weave_prompt")
    pending_pages = {row.get("page") for row in pending_rows}
    pending_weave_jobs = [job for job in weave_jobs.values() if job.page in pending_pages]

    if pending_weave_jobs and not weave_prompt:
        raise ValueError("weave_prompt is not defined but referenced objects were found.")

    failed_pages: set = set()

    async def process_job(job: WeaveJob, worker_id: int) -> None:
        del worker_id
        try:
            prompt = weave_prompt.format(
                ocr=job.base_text,
                objects=format_objects_for_prompt(job.objects),
            )
            woven = await pipeline.infer(prompt, [])
            if bool(settings.get("print_step_outputs", True)):
                tqdm.tqdm.write(msg_debug(f"Weave result: book={book_name} page={job.page} objects={len(job.objects)}"))
            final_outputs[job.page] = woven
        except Exception as exc:
            failed_pages.add(job.page)
            tqdm.tqdm.write(msg_error(f"Weave failed: book={book_name} page={job.page} error={exc}"))
            async with pipeline.write_lock:
                append_jsonl(
                    pipeline.failure_jsonl(file_name),
                    {
                        "stage": "weave",
                        "error": str(exc),
                        "book": book_name,
                        "page": job.page,
                        "source_pages": [obj.source_page for obj in job.objects],
                    },
                )

    if pending_weave_jobs:
        worker_count = min(pipeline.max_in_flight, len(pending_weave_jobs))
        await run_item_worker_pool(
            pending_weave_jobs,
            worker_count=worker_count,
            process_item=process_job,
            progress_desc=msg_info(f"Weave figures [{file_name}]"),
        )

    reference_info: Dict[Any, List[Dict[str, Any]]] = {}
    for assignment in assignments:
        if assignment.target_page is None:
            continue
        reference_info.setdefault(assignment.target_page, []).append(
            {
                "source_page": assignment.source_page,
                "object_index": assignment.object_index,
                "method": assignment.method,
            }
        )

    saved: List[Dict[str, Any]] = []
    for row in sort_rows_by_page(pending_rows):
        page = row.get("page")
        if page in failed_pages:
            continue
        final_row = dict(row)
        final_row["final_output"] = final_outputs.get(page, str(row.get("text", "")))
        final_row["woven_objects"] = reference_info.get(page, [])
        append_jsonl(pipeline.final_jsonl(file_name), final_row)
        saved.append(final_row)

    # resume で追記された行があっても最終ファイルは常に page 順を保証する
    if saved:
        write_jsonl(
            pipeline.final_jsonl(file_name),
            sort_rows_by_page(load_jsonl(pipeline.final_jsonl(file_name))),
        )
    return saved


def load_final_rows(pipeline: Any, file_name: str) -> List[Dict[str, Any]]:
    return load_jsonl(pipeline.final_jsonl(file_name))
