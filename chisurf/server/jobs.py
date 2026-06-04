from __future__ import annotations

import enum
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Job:
    """Represents a single asynchronous operation."""

    def __init__(
        self,
        job_id: str,
        action: str,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Initialise a job.

        Parameters
        ----------
        job_id : str
            Unique job identifier.
        action : str
            Action name for this job.
        params : dict, optional
            Parameters associated with the job.

        """
        self.job_id = job_id
        self.action = action
        self.params = params or {}
        self.status = JobStatus.QUEUED
        self.result: Any = None
        self.error: Optional[str] = None
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self._cancel_event = threading.Event()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the job to a plain dictionary."""
        return {
            "job_id": self.job_id,
            "action": self.action,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class JobManager:
    """Manages lifecycle of async jobs.

    Provides thread-safe job creation, tracking, and basic cancellation
    (cooperative, checked via ``should_cancel``).
    """

    def __init__(self, max_history: int = 100):
        """Initialise an empty job manager.

        Parameters
        ----------
        max_history : int
            Maximum number of terminal jobs kept in memory.

        """
        self._lock = threading.RLock()
        self._jobs: Dict[str, Job] = {}
        self._max_history = max_history

    def create_job(self, action: str, params: Optional[Dict[str, Any]] = None) -> Job:
        """Create a new queued job with a generated UUID.

        Parameters
        ----------
        action : str
            Action name.
        params : dict, optional
            Parameters for the job.

        """
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, action=action, params=params)
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Return a job by its ID, or ``None``.

        Parameters
        ----------
        job_id : str
            Job identifier.

        """
        with self._lock:
            return self._jobs.get(job_id)

    def start_job(self, job_id: str) -> bool:
        """Transition a job from QUEUED to RUNNING.

        Returns ``False`` if the job does not exist or was already
        cancelled.

        Parameters
        ----------
        job_id : str
            Job identifier.

        """
        job = self.get_job(job_id)
        if job is None:
            return False
        with self._lock:
            if job.status == JobStatus.CANCELLED:
                return False
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
        return True

    def complete_job(self, job_id: str, result: Any = None) -> bool:
        """Mark a job as COMPLETED with an optional result.

        Parameters
        ----------
        job_id : str
            Job identifier.
        result : any, optional
            Result value to store.

        """
        job = self.get_job(job_id)
        if job is None:
            return False
        with self._lock:
            job.status = JobStatus.COMPLETED
            job.result = result
            job.finished_at = time.time()
        return True

    def fail_job(self, job_id: str, error: str) -> bool:
        """Mark a job as FAILED with an error message.

        Parameters
        ----------
        job_id : str
            Job identifier.
        error : str
            Error description.

        """
        job = self.get_job(job_id)
        if job is None:
            return False
        with self._lock:
            job.status = JobStatus.FAILED
            job.error = error
            job.finished_at = time.time()
        return True

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job (cooperative, sets a cancellation event).

        Parameters
        ----------
        job_id : str
            Job identifier.

        """
        job = self.get_job(job_id)
        if job is None:
            return False
        with self._lock:
            job.status = JobStatus.CANCELLED
            job._cancel_event.set()
            job.finished_at = time.time()
        return True

    def should_cancel(self, job_id: str) -> bool:
        """Return ``True`` if the job's cancellation event has been set.

        Parameters
        ----------
        job_id : str
            Job identifier.

        """
        job = self.get_job(job_id)
        if job is None:
            return False
        return job._cancel_event.is_set()

    def list_jobs(self, status: Optional[JobStatus] = None) -> List[Job]:
        """Return all jobs, optionally filtered by status.

        Parameters
        ----------
        status : JobStatus, optional
            If given, only jobs with this status are returned.

        """
        with self._lock:
            jobs = list(self._jobs.values())
        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        return jobs

    def cleanup(self) -> int:
        """Remove oldest completed/failed/cancelled jobs beyond ``max_history``."""
        with self._lock:
            terminal = [j for j in self._jobs.values() if j.status in (
                JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED
            )]
            terminal.sort(key=lambda j: j.finished_at or 0.0)
            to_remove = terminal[:-self._max_history] if len(terminal) > self._max_history else []
            for j in to_remove:
                del self._jobs[j.job_id]
        return len(to_remove)

    # ── synchronous execution helpers ───────────────────────────────

    def run_fn(self, action: str, params: Optional[Dict[str, Any]], fn: Callable) -> Job:
        """Synchronously create and execute a job.  Returns the completed job."""
        job = self.create_job(action, params)
        self._execute_job(job, fn)
        return job

    def run_threaded(self, action: str, params: Optional[Dict[str, Any]], fn: Callable) -> Job:
        """Run *fn* in a daemon thread.  Returns the job immediately (RUNNING)."""
        job = self.create_job(action, params)
        t = threading.Thread(
            target=self._execute_job,
            args=(job, fn),
            daemon=True,
        )
        t.start()
        return job

    def _execute_job(self, job: Job, fn: Callable) -> None:
        """Run *fn* inside *job* and record the outcome.

        Parameters
        ----------
        job : Job
            Job to execute.
        fn : callable
            Nullary callable whose return value becomes the job result.

        """
        if job.status == JobStatus.CANCELLED:
            return
        self.start_job(job.job_id)
        try:
            if self.should_cancel(job.job_id):
                return
            result = fn()
            if self.should_cancel(job.job_id):
                self.cancel_job(job.job_id)
            else:
                self.complete_job(job.job_id, result)
        except Exception as e:
            if job.status != JobStatus.CANCELLED:
                self.fail_job(job.job_id, str(e))
