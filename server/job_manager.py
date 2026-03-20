"""Single-worker async job queue for COOL-MC.

Only one job runs at a time — Storm cannot handle concurrent model checking.
"""
from __future__ import annotations

import asyncio
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

WORKDIR = "/workspaces/coolmc"
MLFLOW_TRACKING_URI = f"file://{WORKDIR}/mlruns"
LOGS_DIR = Path(WORKDIR) / "logs"
JOBS_FILE = Path(WORKDIR) / "jobs.json"

LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------

@dataclass
class Job:
    job_id: str
    status: str                         # PENDING | RUNNING | DONE | FAILED | CANCELLED
    args: list[str]
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    pid: Optional[int] = None
    returncode: Optional[int] = None
    mlflow_run_id: Optional[str] = None
    log_path: Optional[str] = None
    queue_position: Optional[int] = None


# ---------------------------------------------------------------------------
# Job manager
# ---------------------------------------------------------------------------

class JobManager:
    def __init__(self):
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        self._jobs: dict[str, Job] = {}
        self._current: Optional[Job] = None
        self._current_proc: Optional[asyncio.subprocess.Process] = None
        self._load_checkpoint()

    # ── Public API ─────────────────────────────────────────────────────────

    def enqueue(self, params: dict) -> Job:
        args = _params_to_args(params)
        job_id = str(uuid.uuid4())
        job = Job(
            job_id=job_id,
            status="PENDING",
            args=args,
            created_at=datetime.utcnow().isoformat(),
            log_path=str(LOGS_DIR / f"{job_id}.log"),
        )
        self._jobs[job.job_id] = job
        self._queue.put_nowait(job)
        self._update_queue_positions()
        self._checkpoint()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        job = self._jobs.get(job_id)
        if job:
            self._update_queue_positions()
        return job

    def list_all(self) -> list[Job]:
        self._update_queue_positions()
        return list(self._jobs.values())

    def queue_status(self) -> dict:
        self._update_queue_positions()
        pending = [j for j in self._jobs.values() if j.status == "PENDING"]
        return {
            "running": self._current.job_id if self._current else None,
            "pending_count": len(pending),
            "pending": [j.job_id for j in pending],
        }

    async def cancel(self, job_id: str) -> Optional[Job]:
        job = self._jobs.get(job_id)
        if not job:
            return None
        if job.status == "PENDING":
            job.status = "CANCELLED"
            self._checkpoint()
            return job
        if job.status == "RUNNING" and self._current_proc:
            self._current_proc.terminate()
            job.status = "CANCELLED"
            job.finished_at = datetime.utcnow().isoformat()
            self._checkpoint()
            return job
        return job

    # ── Worker ─────────────────────────────────────────────────────────────

    async def run_worker(self):
        """Single worker — consumes one job at a time, forever."""
        while True:
            job = await self._queue.get()
            if job.status == "CANCELLED":
                continue

            self._current = job
            job.status = "RUNNING"
            job.started_at = datetime.utcnow().isoformat()
            self._update_queue_positions()
            self._checkpoint()

            await self._run_job(job)

            self._current = None
            self._current_proc = None
            self._checkpoint()

    async def _run_job(self, job: Job):
        log_file = open(job.log_path, "w")
        try:
            proc = await asyncio.create_subprocess_exec(
                "python", "cool_mc.py", *job.args,
                cwd=WORKDIR,
                stdout=log_file,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI},
            )
            job.pid = proc.pid
            self._current_proc = proc
            self._checkpoint()

            await proc.wait()

            job.returncode = proc.returncode
            job.status = "DONE" if proc.returncode == 0 else "FAILED"
            job.finished_at = datetime.utcnow().isoformat()
            job.mlflow_run_id = _extract_run_id(job.log_path)
        except Exception as exc:
            job.status = "FAILED"
            job.finished_at = datetime.utcnow().isoformat()
            log_file.write(f"\n[job_manager] Exception: {exc}\n")
        finally:
            log_file.close()

    # ── Persistence ────────────────────────────────────────────────────────

    def _checkpoint(self):
        data = {jid: asdict(j) for jid, j in self._jobs.items()}
        JOBS_FILE.write_text(json.dumps(data, indent=2))

    def _load_checkpoint(self):
        if not JOBS_FILE.exists():
            return
        try:
            data = json.loads(JOBS_FILE.read_text())
            for jid, d in data.items():
                job = Job(**d)
                # Jobs that were RUNNING when the server crashed → mark FAILED
                if job.status == "RUNNING":
                    job.status = "FAILED"
                    job.finished_at = datetime.utcnow().isoformat()
                self._jobs[jid] = job
        except Exception:
            pass  # corrupt checkpoint — start fresh

    # ── Helpers ────────────────────────────────────────────────────────────

    def _update_queue_positions(self):
        pending = [j for j in self._jobs.values() if j.status == "PENDING"]
        for i, job in enumerate(pending):
            job.queue_position = i + 1
        for job in self._jobs.values():
            if job.status != "PENDING":
                job.queue_position = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _params_to_args(params: dict) -> list[str]:
    """Convert a params dict to a cool_mc.py CLI argument list."""
    args = []
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.extend([f"--{key}", str(value)])
    return args


def _extract_run_id(log_path: str) -> Optional[str]:
    """Try to extract an MLflow run_id from the job log output."""
    try:
        with open(log_path) as f:
            for line in f:
                # MLflow typically prints: "run_id: <hex>"
                if "run_id" in line.lower():
                    parts = line.strip().split()
                    for part in parts:
                        if len(part) == 32 and part.isalnum():
                            return part
    except Exception:
        pass
    return None


# Singleton
manager = JobManager()
