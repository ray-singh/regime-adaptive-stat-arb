"""Minimal async job queue for long-running backtests.

Usage (from app.py)::

    from backtest.job_queue import BacktestJobQueue
    job_queue = BacktestJobQueue(max_workers=2)

    # Submit
    job_id = job_queue.submit(payload)       # returns immediately

    # Poll
    job = job_queue.get(job_id)              # BacktestJob or None
    job.status                               # "pending" | "running" | "complete" | "failed"
    job.result                               # dict when complete
    job.error                                # str when failed

    # List
    job_queue.list_jobs()                    # list[dict] summary of all jobs

Design notes:
    - ThreadPoolExecutor: each job runs in a daemon worker thread.  This is
      appropriate because the GIL is released in the numpy/statsmodels hot-paths
      and no shared mutable state is touched by concurrent jobs.
    - Jobs are stored in-memory (process lifetime only).  On restart they are
      lost — that is intentional for this minimal queue.
    - A soft cap (MAX_HISTORY) prevents unbounded memory growth from completed
      jobs; oldest completed/failed jobs are evicted first.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

MAX_HISTORY = 50   # keep at most this many completed/failed jobs


class JobStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    COMPLETE = "complete"
    FAILED   = "failed"
    CANCELED = "canceled"


@dataclass
class BacktestJob:
    job_id:       str
    payload:      dict
    status:       JobStatus   = JobStatus.PENDING
    submitted_at: str         = ""
    started_at:   Optional[str] = None
    finished_at:  Optional[str] = None
    result:       Optional[dict[str, Any]] = None
    error:        Optional[str] = None
    # Internal: future returned by executor (used for cancellation)
    _future:      Any          = field(default=None, repr=False)

    def to_summary(self) -> dict:
        """Lightweight dict safe for JSON serialisation (no result blob)."""
        return {
            "job_id":       self.job_id,
            "status":       self.status.value,
            "submitted_at": self.submitted_at,
            "started_at":   self.started_at,
            "finished_at":  self.finished_at,
            "error":        self.error,
        }


class BacktestJobQueue:
    """Thread-pool backed queue for async backtest execution.

    Parameters
    ----------
    runner : callable(payload) -> dict
        The function that executes a backtest and returns a result dict.
        Typically ``lambda p: run_backtest(*_build_config_from_payload(p))``.
    max_workers : int
        Maximum concurrent backtest jobs.
    """

    def __init__(self, runner: Callable[[dict], dict], max_workers: int = 2):
        self._runner   = runner
        self._lock     = threading.Lock()
        self._jobs:    dict[str, BacktestJob] = {}
        self._shutdown = False
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="backtest-worker",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, payload: dict) -> str:
        """Enqueue a backtest job.  Returns job_id immediately.

        The future is stored on the job *inside* the same lock acquisition
        that registers the job, ensuring cancel() always sees a valid future.
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("BacktestJobQueue has been shut down")
            job_id = uuid.uuid4().hex[:12]
            job = BacktestJob(
                job_id=job_id,
                payload=payload,
                status=JobStatus.PENDING,
                submitted_at=_utcnow(),
            )
            self._evict_old_jobs()
            self._jobs[job_id] = job
            # Submit inside the lock so _future is set before any concurrent
            # cancel() call can observe this job with _future=None.
            future = self._executor.submit(self._run_job, job_id)
            job._future = future

        logger.info("Job %s submitted (queue depth: %d)", job_id, len(self._jobs))
        return job_id

    def get(self, job_id: str) -> Optional[BacktestJob]:
        """Return the BacktestJob for *job_id*, or None if not found."""
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Attempt to cancel a pending job.  Returns True if cancellation succeeded.

        Running jobs cannot be cancelled (backtest is already in-flight).
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status != JobStatus.PENDING:
                return False
            if job._future is not None:
                cancelled = job._future.cancel()
            else:
                cancelled = True
            if cancelled:
                job.status = JobStatus.CANCELED
                job.finished_at = _utcnow()
                logger.info("Job %s canceled", job_id)
            return cancelled

    def shutdown(self, wait: bool = True) -> None:
        """Gracefully shut down the executor.

        After this call, any further ``submit()`` calls will raise
        ``RuntimeError``.  In-flight jobs are allowed to finish when
        *wait* is True (the default).
        """
        with self._lock:
            self._shutdown = True
        self._executor.shutdown(wait=wait)
        logger.info("BacktestJobQueue shut down (wait=%s)", wait)

    def list_jobs(self, limit: int = 50) -> list[dict]:
        """Return summary dicts for the most-recent *limit* jobs."""
        with self._lock:
            jobs = sorted(
                self._jobs.values(),
                key=lambda j: j.submitted_at,
                reverse=True,
            )[:limit]
            return [j.to_summary() for j in jobs]

    def active_count(self) -> int:
        """Number of jobs currently pending or running."""
        with self._lock:
            return sum(
                1 for j in self._jobs.values()
                if j.status in (JobStatus.PENDING, JobStatus.RUNNING)
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status == JobStatus.CANCELED:
                return
            job.status     = JobStatus.RUNNING
            job.started_at = _utcnow()

        logger.info("Job %s started", job_id)
        try:
            result = self._runner(job.payload)
            with self._lock:
                job.status      = JobStatus.COMPLETE
                job.result      = result
                job.finished_at = _utcnow()
            logger.info("Job %s complete", job_id)
        except Exception as exc:
            with self._lock:
                job.status      = JobStatus.FAILED
                job.error       = str(exc)
                job.finished_at = _utcnow()
            logger.error("Job %s failed: %s", job_id, exc, exc_info=True)

    def _evict_old_jobs(self) -> None:
        """Remove oldest completed/failed jobs when history is full."""
        done = [
            j for j in self._jobs.values()
            if j.status in (JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELED)
        ]
        if len(self._jobs) >= MAX_HISTORY and done:
            done_sorted = sorted(done, key=lambda j: j.submitted_at)
            to_remove = len(self._jobs) - MAX_HISTORY + 1
            for j in done_sorted[:to_remove]:
                del self._jobs[j.job_id]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
