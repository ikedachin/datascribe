from typing import Awaitable, Callable, Iterable, TypeVar

import asyncio

import tqdm


T = TypeVar("T")


async def run_item_worker_pool(
    items: Iterable[T],
    *,
    worker_count: int,
    process_item: Callable[[T, int], Awaitable[None]],
    progress_desc: str | None = None,
) -> None:
    """item 単位の asyncio worker pool。

    progress_desc を渡すと tqdm の進捗バーを表示する（1 item 完了ごとに更新）。
    """
    items = list(items)
    progress = None
    if progress_desc is not None and items:
        progress = tqdm.tqdm(total=len(items), desc=progress_desc, unit="item")

    queue: asyncio.Queue[T | None] = asyncio.Queue(maxsize=max(1, worker_count * 2))
    workers = [
        asyncio.create_task(_item_worker(queue, process_item, worker_id, progress))
        for worker_id in range(worker_count)
    ]

    for item in items:
        await queue.put(item)

    await queue.join()

    for _ in workers:
        await queue.put(None)
    await asyncio.gather(*workers)

    if progress is not None:
        progress.close()


async def _item_worker(
    queue: asyncio.Queue[T | None],
    process_item: Callable[[T, int], Awaitable[None]],
    worker_id: int,
    progress: "tqdm.tqdm | None" = None,
) -> None:
    while True:
        item = await queue.get()
        try:
            if item is None:
                return
            await process_item(item, worker_id)
            if progress is not None:
                progress.update(1)
        finally:
            queue.task_done()
