"""
Async message queue for longer-running tasks (RAG pipeline, eval runs, etc.).

Architecture:
  - Producer adds Task objects to an asyncio.Queue
  - N worker coroutines drain the queue concurrently
  - Results collected in a result dict keyed by task ID
  - Supports priority levels and task timeouts
  - Tracks queue depth, wait time, and processing time per task
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Coroutine


class Priority(IntEnum):
    HIGH   = 1
    NORMAL = 5
    LOW    = 10


@dataclass(order=True)
class Task:
    priority: Priority
    task_id: str = field(compare=False)
    name: str = field(compare=False)
    fn: Callable[[], Coroutine] = field(compare=False)
    enqueued_at: float = field(default_factory=time.time, compare=False)
    timeout_s: float = field(default=30.0, compare=False)


@dataclass
class TaskResult:
    task_id: str
    name: str
    success: bool
    result: Any
    error: str
    wait_ms: float      # time spent in queue
    process_ms: float   # time to execute


class MessageQueue:
    """
    Priority-aware async task queue with N concurrent workers.
    """

    def __init__(self, n_workers: int = 4):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._results: dict[str, TaskResult] = {}
        self._n_workers = n_workers
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._processed = 0
        self._failed = 0

    async def start(self) -> None:
        self._running = True
        self._workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self._n_workers)
        ]

    async def stop(self) -> None:
        self._running = False
        # Send sentinel None for each worker to unblock
        for _ in range(self._n_workers):
            await self._queue.put((0, None))
        await asyncio.gather(*self._workers, return_exceptions=True)

    def submit(
        self,
        fn: Callable[[], Coroutine],
        name: str = "",
        priority: Priority = Priority.NORMAL,
        timeout_s: float = 30.0,
    ) -> str:
        task_id = str(uuid.uuid4())[:8]
        task = Task(
            priority=priority,
            task_id=task_id,
            name=name or task_id,
            fn=fn,
            timeout_s=timeout_s,
        )
        self._queue.put_nowait((priority.value, task))
        return task_id

    def get_result(self, task_id: str) -> TaskResult | None:
        return self._results.get(task_id)

    async def wait_for(self, task_id: str, poll_interval: float = 0.05) -> TaskResult:
        while task_id not in self._results:
            await asyncio.sleep(poll_interval)
        return self._results[task_id]

    async def _worker(self, worker_id: int) -> None:
        while True:
            _, item = await self._queue.get()
            if item is None:
                self._queue.task_done()
                break

            task: Task = item
            wait_ms = round((time.time() - task.enqueued_at) * 1000, 1)
            t0 = time.perf_counter()
            error = ""
            result = None
            success = False

            try:
                result = await asyncio.wait_for(task.fn(), timeout=task.timeout_s)
                success = True
                self._processed += 1
            except asyncio.TimeoutError:
                error = f"Timeout after {task.timeout_s}s"
                self._failed += 1
            except Exception as e:
                error = str(e)
                self._failed += 1

            process_ms = round((time.perf_counter() - t0) * 1000, 1)
            self._results[task.task_id] = TaskResult(
                task_id=task.task_id,
                name=task.name,
                success=success,
                result=result,
                error=error,
                wait_ms=wait_ms,
                process_ms=process_ms,
            )
            self._queue.task_done()

    def stats(self) -> dict:
        return {
            "queue_depth": self._queue.qsize(),
            "n_workers": self._n_workers,
            "processed": self._processed,
            "failed": self._failed,
            "results_stored": len(self._results),
        }


# ---------------------------------------------------------------------------
# Convenience: run a batch of tasks and collect all results
# ---------------------------------------------------------------------------

async def run_batch(
    tasks: list[tuple[str, Callable[[], Coroutine]]],
    n_workers: int = 4,
    priority: Priority = Priority.NORMAL,
) -> dict[str, TaskResult]:
    """Submit a batch of (name, async_fn) tasks, drain the queue, return all results."""
    q = MessageQueue(n_workers=n_workers)
    await q.start()

    ids = {}
    for name, fn in tasks:
        tid = q.submit(fn, name=name, priority=priority)
        ids[name] = tid

    await q._queue.join()
    await q.stop()

    return {name: q.get_result(tid) for name, tid in ids.items()}
