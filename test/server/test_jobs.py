from __future__ import annotations

import time
import threading
from chisurf.server.jobs import JobManager, JobStatus


class TestJobManager:
    """Async job manager for long-running operations."""

    def test_create_and_get_job(self):
        mgr = JobManager()
        job = mgr.create_job("fit.run", {"fit_index": 0})
        assert job.job_id is not None
        assert job.action == "fit.run"
        assert job.params == {"fit_index": 0}
        assert job.status == JobStatus.QUEUED

        retrieved = mgr.get_job(job.job_id)
        assert retrieved is job

    def test_get_nonexistent_job(self):
        mgr = JobManager()
        assert mgr.get_job("nonexistent") is None

    def test_start_job(self):
        mgr = JobManager()
        job = mgr.create_job("fit.run", {})
        mgr.start_job(job.job_id)
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None

    def test_complete_job(self):
        mgr = JobManager()
        job = mgr.create_job("fit.run", {})
        mgr.start_job(job.job_id)
        mgr.complete_job(job.job_id, {"chi2": 1.2})
        assert job.status == JobStatus.COMPLETED
        assert job.result == {"chi2": 1.2}
        assert job.finished_at is not None

    def test_fail_job(self):
        mgr = JobManager()
        job = mgr.create_job("fit.run", {})
        mgr.start_job(job.job_id)
        mgr.fail_job(job.job_id, "Something went wrong")
        assert job.status == JobStatus.FAILED
        assert job.error == "Something went wrong"

    def test_cancel_job_queued(self):
        mgr = JobManager()
        job = mgr.create_job("fit.run", {})
        mgr.cancel_job(job.job_id)
        assert job.status == JobStatus.CANCELLED

    def test_cancel_job_running(self):
        mgr = JobManager()
        job = mgr.create_job("fit.run", {})
        mgr.start_job(job.job_id)
        mgr.cancel_job(job.job_id)
        assert job.status == JobStatus.CANCELLED

    def test_cancel_nonexistent(self):
        mgr = JobManager()
        mgr.cancel_job("nonexistent")  # must not raise

    def test_should_cancel(self):
        mgr = JobManager()
        job = mgr.create_job("fit.run", {})
        mgr.start_job(job.job_id)
        assert not mgr.should_cancel(job.job_id)
        mgr.cancel_job(job.job_id)
        assert mgr.should_cancel(job.job_id)

    def test_should_cancel_nonexistent(self):
        mgr = JobManager()
        assert not mgr.should_cancel("nonexistent")

    def test_list_jobs(self):
        mgr = JobManager()
        j1 = mgr.create_job("fit.run", {})
        j2 = mgr.create_job("fit.run", {})
        jobs = mgr.list_jobs()
        assert len(jobs) == 2
        assert jobs[0].job_id == j1.job_id
        assert jobs[1].job_id == j2.job_id

    def test_list_jobs_filter_status(self):
        mgr = JobManager()
        j1 = mgr.create_job("fit.run", {})
        j2 = mgr.create_job("fit.run", {})
        mgr.start_job(j2.job_id)
        queued = mgr.list_jobs(status=JobStatus.QUEUED)
        running = mgr.list_jobs(status=JobStatus.RUNNING)
        assert len(queued) == 1
        assert queued[0].job_id == j1.job_id
        assert len(running) == 1
        assert running[0].job_id == j2.job_id

    def test_cleanup_removes_old_completed_jobs(self):
        mgr = JobManager(max_history=2)
        j1 = mgr.create_job("a", {})
        j2 = mgr.create_job("b", {})
        j3 = mgr.create_job("c", {})

        # Mark j1 as completed
        mgr.start_job(j1.job_id)
        mgr.complete_job(j1.job_id, {})

        mgr.cleanup()
        # Should keep j2 and j3 (they are more recent), and j1 (completed but only 1 completed so far)
        assert mgr.get_job(j1.job_id) is not None

        # Complete one more and cleanup
        mgr.start_job(j2.job_id)
        mgr.complete_job(j2.job_id, {})
        mgr.start_job(j3.job_id)
        mgr.complete_job(j3.job_id, {})
        mgr.cleanup()

        # Oldest completed (j1) should be removed
        assert mgr.get_job(j1.job_id) is None

    def test_run_fn_queues_and_completes(self):
        """run_fn should create a job, execute fn, and mark completed."""
        mgr = JobManager()

        def my_task():
            return 42

        job = mgr.run_fn("calculate", {}, my_task)
        assert job.status == JobStatus.COMPLETED
        assert job.result == 42

    def test_run_fn_propagates_exception(self):
        mgr = JobManager()

        def broken():
            raise ValueError("oops")

        job = mgr.run_fn("calculate", {}, broken)
        assert job.status == JobStatus.FAILED
        assert "oops" in (job.error or "")

    def test_run_fn_cancellation(self):
        """If cancelled before execution, the fn should not run."""
        mgr = JobManager()
        ran = threading.Event()

        def my_task():
            ran.set()
            return 42

        job = mgr.create_job("calculate", {})
        mgr.cancel_job(job.job_id)
        mgr._execute_job(job, my_task)
        assert not ran.is_set()
        assert job.status == JobStatus.CANCELLED

    def test_run_threaded(self):
        """run_threaded should run fn in a background thread."""
        mgr = JobManager()
        result_holder = []

        def my_task():
            time.sleep(0.01)
            return "done"

        job = mgr.run_threaded("calculate", {}, my_task)
        assert job.status == JobStatus.RUNNING

        # Wait for completion
        timeout = 5.0
        while job.status == JobStatus.RUNNING and timeout > 0:
            time.sleep(0.05)
            timeout -= 0.05

        assert job.status == JobStatus.COMPLETED
        assert job.result == "done"
