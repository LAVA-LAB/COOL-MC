"""COOL-MC FastAPI server — runs inside the Docker container."""
from __future__ import annotations

import asyncio
import io
import zipfile
from pathlib import Path
from typing import Any, Optional

from mlflow.store.tracking.file_store import FileStore
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

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


@app.get("/info")
def server_info() -> dict:
    """Return COOL-MC version info and service URLs."""
    import subprocess
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=WORKDIR, text=True
        ).strip()
        commit_date = subprocess.check_output(
            ["git", "log", "-1", "--format=%ai"], cwd=WORKDIR, text=True
        ).strip()
    except Exception:
        commit = "unknown"
        commit_date = "unknown"
    return {
        "server": "COOL-MC FastAPI Server",
        "cool_mc_commit": commit,
        "cool_mc_commit_date": commit_date,
        "mlflow_ui": "http://localhost:5000",
    }


# ---------------------------------------------------------------------------
# MLflow query endpoints
# ---------------------------------------------------------------------------

@app.get("/mlflow/experiments")
def list_experiments() -> list[dict]:
    """List all MLflow experiments."""
    import mlflow
    mlflow.set_tracking_uri(f"file://{WORKDIR}/mlruns")
    client = mlflow.tracking.MlflowClient()
    return [
        {"experiment_id": e.experiment_id, "name": e.name, "artifact_location": e.artifact_location}
        for e in client.search_experiments()
    ]


@app.get("/mlflow/runs")
def list_runs(experiment_id: str = "0", max_results: int = 50) -> list[dict]:
    """List MLflow runs for an experiment, newest first."""
    import mlflow
    mlflow.set_tracking_uri(f"file://{WORKDIR}/mlruns")
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        max_results=max_results,
        order_by=["attribute.start_time DESC"],
    )
    return [_run_dict(r) for r in runs]


@app.get("/mlflow/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """Get a single MLflow run by ID."""
    import mlflow
    mlflow.set_tracking_uri(f"file://{WORKDIR}/mlruns")
    client = mlflow.tracking.MlflowClient()
    try:
        run = client.get_run(run_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_dict(run)


@app.get("/mlflow/runs/{run_id}/metrics/{metric_key}")
def get_metric_history(run_id: str, metric_key: str) -> list[dict]:
    """Get the full history of a metric for a run."""
    import mlflow
    mlflow.set_tracking_uri(f"file://{WORKDIR}/mlruns")
    client = mlflow.tracking.MlflowClient()
    try:
        history = client.get_metric_history(run_id, metric_key)
    except Exception:
        raise HTTPException(status_code=404, detail="Metric not found")
    return [{"step": m.step, "value": m.value, "timestamp": m.timestamp} for m in history]


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


@app.post("/jobs/cancel-all")
def cancel_all_jobs() -> list[dict]:
    """Cancel all pending and running jobs."""
    return [_job_dict(j) for j in manager.cancel_all()]


@app.delete("/jobs/history")
def clear_job_history() -> dict:
    """Remove all completed (DONE/FAILED/CANCELLED) jobs from history."""
    count = manager.clear_history()
    return {"cleared": count}


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


@app.post("/files/prism-bundle")
async def upload_prism_bundle(file: UploadFile = File(...)) -> dict:
    """Upload a zip archive containing a .prism file and any companion directories.

    The zip is extracted directly into prism_files_user/, preserving the internal
    directory structure. Use this for environments like icu_sepsis that ship with
    a subfolder of data files alongside the .prism file.
    """
    data = await file.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        prism_names = [n for n in zf.namelist() if n.endswith(".prism")]
        if not prism_names:
            raise HTTPException(status_code=400, detail="Zip contains no .prism file")
        zf.extractall(_PRISM_USER_DIR)
    return {
        "extracted": zf.namelist(),
        "prism_files": [f"prism_files_user/{n}" for n in prism_names],
    }


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
# Storm / stormpy endpoints
# ---------------------------------------------------------------------------

class _StormRequest(BaseModel):
    prism_file_path: str
    constant_definitions: str = ""
    formula: str = ""
    all_states: bool = False
    nr_steps: int = 100
    seed: Optional[int] = None


def _storm_endpoint(fn, req: _StormRequest, **extra):
    """Shared error-handling wrapper for Storm endpoints."""
    try:
        return fn(**extra)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/storm/build-model")
def storm_build_model(req: _StormRequest) -> dict:
    """Build a stormpy model and return statistics (type, states, transitions, labels)."""
    from server.storm_api import build_model
    return _storm_endpoint(
        build_model, req,
        prism_file_path=req.prism_file_path,
        constant_definitions=req.constant_definitions,
        formula=req.formula,
    )


@app.post("/storm/check")
def storm_check(req: _StormRequest) -> dict:
    """Model-check a formula and return the result for the initial state."""
    if not req.formula:
        raise HTTPException(status_code=422, detail="formula is required")
    from server.storm_api import check
    return _storm_endpoint(
        check, req,
        prism_file_path=req.prism_file_path,
        formula=req.formula,
        constant_definitions=req.constant_definitions,
        all_states=req.all_states,
    )


@app.post("/storm/get-scheduler")
def storm_get_scheduler(req: _StormRequest) -> dict:
    """Extract an optimal scheduler via model checking."""
    if not req.formula:
        raise HTTPException(status_code=422, detail="formula is required")
    from server.storm_api import get_scheduler
    return _storm_endpoint(
        get_scheduler, req,
        prism_file_path=req.prism_file_path,
        formula=req.formula,
        constant_definitions=req.constant_definitions,
    )


@app.post("/storm/simulate")
def storm_simulate(req: _StormRequest) -> dict:
    """Run a Monte Carlo simulation on a PRISM model."""
    from server.storm_api import simulate
    return _storm_endpoint(
        simulate, req,
        prism_file_path=req.prism_file_path,
        nr_steps=req.nr_steps,
        constant_definitions=req.constant_definitions,
        seed=req.seed,
    )


@app.post("/storm/parametric-check")
def storm_parametric_check(req: _StormRequest) -> dict:
    """Perform parametric model checking (returns rational function)."""
    if not req.formula:
        raise HTTPException(status_code=422, detail="formula is required")
    from server.storm_api import parametric_check
    return _storm_endpoint(
        parametric_check, req,
        prism_file_path=req.prism_file_path,
        formula=req.formula,
        constant_definitions=req.constant_definitions,
    )


@app.post("/storm/transitions")
def storm_get_transitions(req: _StormRequest) -> dict:
    """Return the full sparse transition matrix of a PRISM model."""
    from server.storm_api import get_transitions
    return _storm_endpoint(
        get_transitions, req,
        prism_file_path=req.prism_file_path,
        constant_definitions=req.constant_definitions,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _job_dict(job: Job) -> dict:
    from dataclasses import asdict
    return asdict(job)


def _run_dict(run) -> dict:
    return {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
    }
