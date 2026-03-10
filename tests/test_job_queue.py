"""Unit tests for BacktestJobQueue — covering all stability scenarios."""
import threading
import time

import pytest

from backtest.job_queue import BacktestJobQueue, JobStatus, MAX_HISTORY


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fast_runner(payload: dict) -> dict:
    """Instant success runner."""
    return {"ok": True, "payload": payload}


def _slow_runner(payload: dict) -> dict:
    """Blocks until payload['event'] is set, then returns."""
    payload["event"].wait(timeout=5)
    return {"ok": True}


def _failing_runner(payload: dict) -> dict:
    raise ValueError("simulated backtest failure")


def _wait_for_status(queue, job_id, target_status, timeout=3.0):
    """Poll until job reaches target_status or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = queue.get(job_id)
        if job and job.status == target_status:
            return job
        time.sleep(0.01)
    return queue.get(job_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSubmitGet:
    def test_submit_returns_job_id(self):
        q = BacktestJobQueue(runner=_fast_runner, max_workers=1)
        job_id = q.submit({"x": 1})
        assert isinstance(job_id, str) and len(job_id) == 12
        q.shutdown()

    def test_get_returns_none_for_unknown(self):
        q = BacktestJobQueue(runner=_fast_runner, max_workers=1)
        assert q.get("nonexistent_id") is None
        q.shutdown()

    def test_submitted_job_eventually_completes(self):
        q = BacktestJobQueue(runner=_fast_runner, max_workers=1)
        job_id = q.submit({"x": 1})
        job = _wait_for_status(q, job_id, JobStatus.COMPLETE)
        assert job is not None
        assert job.status == JobStatus.COMPLETE
        assert job.result == {"ok": True, "payload": {"x": 1}}
        assert job.started_at is not None
        assert job.finished_at is not None
        q.shutdown()

    def test_failed_job_records_error(self):
        q = BacktestJobQueue(runner=_failing_runner, max_workers=1)
        job_id = q.submit({})
        job = _wait_for_status(q, job_id, JobStatus.FAILED)
        assert job.status == JobStatus.FAILED
        assert "simulated backtest failure" in job.error
        assert job.finished_at is not None
        q.shutdown()


class TestCancel:
    def test_cancel_nonexistent_returns_false(self):
        q = BacktestJobQueue(runner=_fast_runner, max_workers=1)
        assert q.cancel("no_such_id") is False
        q.shutdown()

    def test_cancel_pending_job(self):
        """Fill the single worker thread so the second job stays PENDING."""
        blocker_event = threading.Event()
        results = []

        def blocking_runner(payload):
            blocker_event.wait(timeout=5)
            results.append("unblocked")
            return {}

        q = BacktestJobQueue(runner=blocking_runner, max_workers=1)
        # This job blocks the one worker
        q.submit({"blocker": True, "event": blocker_event})
        time.sleep(0.05)  # let the first job start

        # This job will be PENDING (worker busy)
        job2_id = q.submit({})

        job2 = q.get(job2_id)
        # The second job should still be PENDING
        assert job2.status == JobStatus.PENDING

        cancelled = q.cancel(job2_id)
        assert cancelled is True
        assert q.get(job2_id).status == JobStatus.CANCELED

        blocker_event.set()
        q.shutdown()

    def test_cancel_running_job_returns_false(self):
        """Running jobs cannot be cancelled."""
        running_event = threading.Event()
        blocker_event = threading.Event()

        def signalling_runner(payload):
            running_event.set()
            blocker_event.wait(timeout=5)
            return {}

        q = BacktestJobQueue(runner=signalling_runner, max_workers=1)
        job_id = q.submit({"event": blocker_event})

        # Wait until job is actually running
        running_event.wait(timeout=3)
        assert q.get(job_id).status == JobStatus.RUNNING

        result = q.cancel(job_id)
        assert result is False

        blocker_event.set()
        q.shutdown()

    def test_cancel_race_future_always_set(self):
        """Critical: _future must be set before job is visible to cancel()."""
        # Verify the fix: after submit() returns, _future is never None for PENDING jobs.
        blocker_event = threading.Event()

        def blocking_runner(payload):
            blocker_event.wait(timeout=5)
            return {}

        q = BacktestJobQueue(runner=blocking_runner, max_workers=1)
        q.submit({"blocker": True, "event": blocker_event})
        time.sleep(0.05)  # let first job start

        job2_id = q.submit({})

        # Immediately after submit(), _future must be set on the pending job
        job2 = q.get(job2_id)
        assert job2 is not None
        assert job2._future is not None, (
            "_future must be set atomically inside submit() to prevent cancel() race"
        )

        blocker_event.set()
        q.shutdown()


class TestListAndActiveCount:
    def test_list_jobs_empty(self):
        q = BacktestJobQueue(runner=_fast_runner, max_workers=1)
        assert q.list_jobs() == []
        q.shutdown()

    def test_list_jobs_contains_submitted(self):
        q = BacktestJobQueue(runner=_fast_runner, max_workers=1)
        job_id = q.submit({"n": 1})
        _wait_for_status(q, job_id, JobStatus.COMPLETE)
        summaries = q.list_jobs()
        ids = [s["job_id"] for s in summaries]
        assert job_id in ids
        q.shutdown()

    def test_list_jobs_limit(self):
        q = BacktestJobQueue(runner=_fast_runner, max_workers=4)
        ids = [q.submit({}) for _ in range(10)]
        for jid in ids:
            _wait_for_status(q, jid, JobStatus.COMPLETE)
        assert len(q.list_jobs(limit=3)) == 3
        q.shutdown()

    def test_active_count_zero_after_completion(self):
        q = BacktestJobQueue(runner=_fast_runner, max_workers=1)
        job_id = q.submit({})
        _wait_for_status(q, job_id, JobStatus.COMPLETE)
        assert q.active_count() == 0
        q.shutdown()


class TestEviction:
    def test_eviction_keeps_history_within_max(self):
        q = BacktestJobQueue(runner=_fast_runner, max_workers=4)
        # Submit enough jobs to trigger eviction
        ids = []
        for i in range(MAX_HISTORY + 5):
            ids.append(q.submit({"i": i}))

        # Wait for all to complete
        for jid in ids:
            _wait_for_status(q, jid, JobStatus.COMPLETE, timeout=10)

        # Submit one more to trigger eviction check
        extra_id = q.submit({})
        _wait_for_status(q, extra_id, JobStatus.COMPLETE)

        with q._lock:
            assert len(q._jobs) <= MAX_HISTORY
        q.shutdown()


class TestShutdown:
    def test_shutdown_prevents_new_submissions(self):
        q = BacktestJobQueue(runner=_fast_runner, max_workers=1)
        q.shutdown()
        with pytest.raises(RuntimeError, match="shut down"):
            q.submit({})

    def test_shutdown_wait_drains_inflight(self):
        """Jobs submitted before shutdown should complete before shutdown returns."""
        unblocked = []
        ready = threading.Event()

        def slow_runner(payload):
            ready.set()
            time.sleep(0.05)
            unblocked.append(True)
            return {}

        q = BacktestJobQueue(runner=slow_runner, max_workers=1)
        job_id = q.submit({})
        ready.wait(timeout=3)          # ensure job is running
        q.shutdown(wait=True)          # should block until job finishes
        assert len(unblocked) == 1, "shutdown(wait=True) must drain in-flight jobs"
        assert q.get(job_id).status == JobStatus.COMPLETE

    def test_double_shutdown_is_safe(self):
        q = BacktestJobQueue(runner=_fast_runner, max_workers=1)
        q.shutdown()
        # Second shutdown should not raise
        q.shutdown()
