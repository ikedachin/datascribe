# Under Constructions

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

NUM_STEPS = 5
worker_count = 5

@dataclass
class PipelineJob:
    book: int
    step: int
    ocr_text: str
    objects_bbox: List[List, Any]
    object_content: str
    ocr_generator: str
    output_text: str
    item_id: Any
    payload: Dict[str, Any]
    previous_outputs: Dict[str, Any] = field(default_factory=dict)

#   - `book`, `page`
#   - `text`（OCR結果）
#   - `objects_bbox`
#   - `object_content`
#   - `duplicate_check`
#   - `ocr_generator`

async def run_one_step(job: PipelineJob, worker_id):
    """
    job.stepによって実行する処理を変える関数
    長くなるので概念だけ
    """
    output = dict(job.previous_outputs)
    
    if job.step == 1:
        # OCR
        print(f"Worker {worker_id} processing {job.item_id} step 1 OCR") 
        response = "ocr result from vlm"
        output["ocr"] = response
    elif job.step == 2:
        # 図、グラフ、表の抽出
        print(f"Worker {worker_id} processing {job.item_id} step 2 Munipurate Graphs,Tables,Pictures")
        pass
    elif job.step == 3:
        print(f"Worker {worker_id} processing {job.item_id} step 3 Understanding Graphs,Tables,Pictures")
        # 図、グラフの、表の理解
        pass
    elif job.step == 4:
        # OCRと図、グラフの内容の結合
        print(f"Worker {worker_id} processing {job.item_id} step 4 Merge infomations to OCR result")
        pass
    elif job.step == 5:
        # 整合性評価
        pass
    else:
        # 例外処理
        pass

    return PipelineJob(
        item_id=job.item_id,
        step=job.step + 1, # ここで、ステップが更新される
        payload=job.payload,
        previous_outputs=output,
    )

async def create_qa_item_pool(texts: List[str]) -> List[Dict[str, Any]]:
    queue = asyncio.Queue(maxsize=worker_count * 2) # 読み込むキュー（シードデータ）の数を指定（メモリを抑えるため）

    async def item_processer(worker_id):
        while True:
            job = await queue.get() # queueに保存したジョブを取り出して実行
    
            # ジョブが無くなったら終了処理
            if job is None:
                queue.task_done()
                return
            try:
                curr = job
                # 同じワーカーがそのアイテムを最後まで処理する
                while curr.step <= NUM_STEPS:
                    curr = await run_one_step(curr, worker_id)
                # curr.step == NUM_STEPS + 1、outputs は最終結果を含む
                results[curr.item_id] = curr.previous_outputs
            except Exception as exc:
                print(f"Worker {worker_id} failed item {job.item_id}: {exc}")
            finally:
                queue.task_done()
            
    # item_processerを実行可能なワーカーを作る
    workers = [asyncio.create_task(item_processer(i)) for i in range(worker_count)]

    def build_initial(item_id: int) -> PipelineJob:
        return PipelineJob(item_id=item_id, step=1, payload={"text": texts[item_id]})

    # キューにシードデータを渡す。maxsizeを超えた分は読み込まれない
    total = len(texts)
    for i in range(total):
        await queue.put(build_initial(i))

    # 実行完了まで待つ
    await queue.join()

    # すべて完了したら、queueにNoneを加え、すべてのworkerを終了させる
    for _ in workers:
        await queue.put(None)
    await asyncio.gather(*workers)

    # return results in input order
    return [results.get(i, {}) for i in range(total)]

if __name__ == "__main__":
    # シードデータのイメージ
    texts = [
        "テキスト1",
        "テキスト2",
        "テキスト3",
        "テキスト4",
        "テキスト5",
        "テキスト6",
        "テキスト7",
        "テキスト8",
        "テキスト9",
        "テキスト10",
        "テキスト11",
        "テキスト12",
        "テキスト13",
        "テキスト14",
        "テキスト15"
    ]

    # 実行
    results = asyncio.run(create_qa_item_pool(texts))
    print(json.dumps(results, ensure_ascii=False, indent=2))
