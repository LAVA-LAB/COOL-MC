"""COOL-MC FastAPI server — runs inside the Docker container."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from mlflow.store.tracking.file_store import FileStore
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse

from server.job_manager import Job, manager, WORKDIR

app = FastAPI(title="COOL-MC Server")


# ---------------------------------------------------------------------------
# Startup: initialise MLflow store + launch the single worker
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _start_worker():
    # Initialise the MLflow file store — this creates mlruns/0/ (the default
    # experiment) if the bind-mounted volume is empty.
    FileStore(f"{WORKDIR}/mlruns")

    asyncio.create_task(manager.run_worker())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/jobs/run", response_model=None)
def run_job(params: dict[str, Any]) -> dict:
    """Enqueue a cool_mc.py invocation. Returns immediately with job_id."""
    job = manager.enqueue(params)
    return _job_dict(job)


@app.get("/jobs/")
def list_jobs() -> list[dict]:
    return [_job_dict(j) for j in manager.list_all()]


@app.get("/queue")
def queue_status() -> dict:
    return manager.queue_status()


@app.get("/jobs/last")
def last_job() -> dict:
    """Return the most recently completed (DONE) job, or 404 if none exists yet."""
    done = [j for j in manager.list_all() if j.status == "DONE" and j.mlflow_run_id]
    if not done:
        raise HTTPException(status_code=404, detail="No completed jobs with an mlflow_run_id yet")
    latest = max(done, key=lambda j: j.finished_at or "")
    return _job_dict(latest)


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_dict(job)


@app.get("/jobs/{job_id}/logs", response_class=PlainTextResponse)
def get_logs(job_id: str) -> str:
    job = manager.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    log_path = Path(job.log_path)
    if not log_path.exists():
        return ""
    return log_path.read_text()


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str) -> dict:
    job = await manager.cancel(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _job_dict(job)


# ---------------------------------------------------------------------------
# PRISM file management
# ---------------------------------------------------------------------------

_PRISM_USER_DIR = Path(f"{WORKDIR}/prism_files_user")

@app.post("/files/prism")
async def upload_prism(file: UploadFile = File(...)) -> dict:
    """Upload a .prism file into the container's prism_files_user/ directory."""
    if not file.filename.endswith(".prism"):
        raise HTTPException(status_code=400, detail="Only .prism files are accepted")
    dest = _PRISM_USER_DIR / file.filename
    dest.write_bytes(await file.read())
    return {"filename": file.filename, "path": f"prism_files_user/{file.filename}"}


@app.get("/files/prism")
def list_prism_files() -> list[str]:
    """List all uploaded user PRISM files."""
    return sorted(p.name for p in _PRISM_USER_DIR.glob("*.prism"))


@app.delete("/files/prism/{filename}")
def delete_prism_file(filename: str) -> dict:
    target = _PRISM_USER_DIR / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")
    target.unlink()
    return {"deleted": filename}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job_dict(job: Job) -> dict:
    from dataclasses import asdict
    return asdict(job)
