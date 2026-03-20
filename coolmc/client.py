"""COOL-MC Python client."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

from .config import SERVER_URL, DEFAULTS


# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------

@dataclass
class Job:
    job_id: str
    status: str                        # PENDING | RUNNING | DONE | FAILED | CANCELLED
    queue_position: Optional[int] = None
    pid: Optional[int] = None
    returncode: Optional[int] = None
    mlflow_run_id: Optional[str] = None
    log_path: Optional[str] = None
    args: list[str] = field(default_factory=list)

    @classmethod
    def _from_dict(cls, d: dict) -> "Job":
        return cls(
            job_id=d["job_id"],
            status=d["status"],
            queue_position=d.get("queue_position"),
            pid=d.get("pid"),
            returncode=d.get("returncode"),
            mlflow_run_id=d.get("mlflow_run_id"),
            log_path=d.get("log_path"),
            args=d.get("args", []),
        )

    def __repr__(self) -> str:
        pos = f" (queue pos {self.queue_position})" if self.queue_position else ""
        run = f" run_id={self.mlflow_run_id}" if self.mlflow_run_id else ""
        return f"<Job {self.job_id[:8]} {self.status}{pos}{run}>"


# ---------------------------------------------------------------------------
# CoolMC client
# ---------------------------------------------------------------------------

class CoolMC:
    """Thin client for the COOL-MC FastAPI server running inside Docker."""

    def __init__(self, server_url: str = SERVER_URL):
        self._url = server_url.rstrip("/")

    # ------------------------------------------------------------------
    # Primary interface
    # ------------------------------------------------------------------

    def cmd(
        self,
        *,
        # ── Meta ──────────────────────────────────────────────────────
        task: str = DEFAULTS["task"],
        project_name: str = DEFAULTS["project_name"],
        parent_run_id: str = DEFAULTS["parent_run_id"],
        prism_dir: str = DEFAULTS["prism_dir"],
        prism_file_path: str = DEFAULTS["prism_file_path"],
        constant_definitions: str = DEFAULTS["constant_definitions"],
        disabled_features: str = DEFAULTS["disabled_features"],
        seed: int = DEFAULTS["seed"],
        training_threshold: float = DEFAULTS["training_threshold"],
        # ── Training ──────────────────────────────────────────────────
        num_episodes: int = DEFAULTS["num_episodes"],
        eval_interval: int = DEFAULTS["eval_interval"],
        sliding_window_size: int = DEFAULTS["sliding_window_size"],
        reward_flag: int = DEFAULTS["reward_flag"],
        max_steps: int = DEFAULTS["max_steps"],
        wrong_action_penalty: int = DEFAULTS["wrong_action_penalty"],
        deploy: int = DEFAULTS["deploy"],
        # ── Behavioral Cloning ────────────────────────────────────────
        bc_epochs: int = DEFAULTS["bc_epochs"],
        accuracy_threshold: float = DEFAULTS["accuracy_threshold"],
        behavioral_cloning: str = DEFAULTS["behavioral_cloning"],
        # ── Preprocessor / Postprocessor / Interpreter / Labeler ──────
        preprocessor: str = DEFAULTS["preprocessor"],
        postprocessor: str = DEFAULTS["postprocessor"],
        interpreter: str = DEFAULTS["interpreter"],
        state_labeler: str = DEFAULTS["state_labeler"],
        # ── Model Checking ────────────────────────────────────────────
        prop: str = DEFAULTS["prop"],
        range_plotting: int = DEFAULTS["range_plotting"],
        # ── Agent ─────────────────────────────────────────────────────
        algorithm: str = DEFAULTS["algorithm"],
        alpha: float = DEFAULTS["alpha"],
        noise_scale: float = DEFAULTS["noise_scale"],
        layers: int = DEFAULTS["layers"],
        neurons: int = DEFAULTS["neurons"],
        replay_buffer_size: int = DEFAULTS["replay_buffer_size"],
        epsilon: float = DEFAULTS["epsilon"],
        epsilon_dec: float = DEFAULTS["epsilon_dec"],
        epsilon_min: float = DEFAULTS["epsilon_min"],
        gamma: float = DEFAULTS["gamma"],
        replace: int = DEFAULTS["replace"],
        lr: float = DEFAULTS["lr"],
        batch_size: int = DEFAULTS["batch_size"],
    ) -> Job:
        """Submit a cool_mc.py invocation. All parameters mirror the CLI flags exactly.

        Returns a Job immediately (status=PENDING or RUNNING). The job is
        queued server-side — only one job runs at a time due to Storm constraints.

        Example::

            job = mc.cmd(
                task="safe_training",
                prism_file_path="frozen_lake.prism",
                algorithm="dqn_agent",
                num_episodes=500,
                project_name="my_experiment",
            )
            job = mc.wait(job.job_id)
            print(job.mlflow_run_id)
        """
        if parent_run_id == "last":
            parent_run_id = self._resolve_last_run_id()

        params: dict[str, Any] = {
            "task": task,
            "project_name": project_name,
            "parent_run_id": parent_run_id,
            "prism_dir": prism_dir,
            "prism_file_path": prism_file_path,
            "constant_definitions": constant_definitions,
            "disabled_features": disabled_features,
            "seed": seed,
            "training_threshold": training_threshold,
            "num_episodes": num_episodes,
            "eval_interval": eval_interval,
            "sliding_window_size": sliding_window_size,
            "reward_flag": reward_flag,
            "max_steps": max_steps,
            "wrong_action_penalty": wrong_action_penalty,
            "deploy": deploy,
            "bc_epochs": bc_epochs,
            "accuracy_threshold": accuracy_threshold,
            "behavioral_cloning": behavioral_cloning,
            "preprocessor": preprocessor,
            "postprocessor": postprocessor,
            "interpreter": interpreter,
            "state_labeler": state_labeler,
            "prop": prop,
            "range_plotting": range_plotting,
            "algorithm": algorithm,
            "alpha": alpha,
            "noise_scale": noise_scale,
            "layers": layers,
            "neurons": neurons,
            "replay_buffer_size": replay_buffer_size,
            "epsilon": epsilon,
            "epsilon_dec": epsilon_dec,
            "epsilon_min": epsilon_min,
            "gamma": gamma,
            "replace": replace,
            "lr": lr,
            "batch_size": batch_size,
        }
        resp = self._post("/jobs/run", params)
        return Job._from_dict(resp)

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def status(self, job_id: str) -> Job:
        """Get current status of a job."""
        return Job._from_dict(self._get(f"/jobs/{job_id}"))

    def wait(self, job_id: str, poll_interval: float = 3.0) -> Job:
        """Block until the job reaches DONE, FAILED, or CANCELLED."""
        while True:
            job = self.status(job_id)
            if job.status in ("DONE", "FAILED", "CANCELLED"):
                return job
            time.sleep(poll_interval)

    def get_logs(self, job_id: str) -> str:
        """Return the full log output for a job (works while RUNNING too)."""
        resp = requests.get(f"{self._url}/jobs/{job_id}/logs", timeout=10)
        resp.raise_for_status()
        return resp.text

    def cancel(self, job_id: str) -> Job:
        """Cancel a PENDING job or kill a RUNNING job."""
        resp = requests.delete(f"{self._url}/jobs/{job_id}", timeout=10)
        resp.raise_for_status()
        return Job._from_dict(resp.json())

    def list_jobs(self) -> list[Job]:
        """List all jobs (pending, running, and history)."""
        return [Job._from_dict(j) for j in self._get("/jobs/")]

    def queue_status(self) -> dict:
        """Return current queue depth and the running job (if any)."""
        return self._get("/queue")

    # ------------------------------------------------------------------
    # PRISM file management
    # ------------------------------------------------------------------

    def upload_prism(self, local_path: str, dest_name: str | None = None) -> str:
        """Upload a local .prism file to the container.

        Args:
            local_path: Path to the local .prism file.
            dest_name:  Optional filename to use inside the container.
                        Defaults to the original filename.

        Returns the path string to pass as ``prism_file_path`` in ``mc.cmd()``.

        Example::

            mc.upload_prism("avoid.prism", dest_name="dummy.prism")
            job = mc.cmd(prism_file_path="prism_files_user/dummy.prism", ...)
        """
        path = Path(local_path)
        # Use dest_name as the filename on the server; fall back to the local filename.
        upload_name = dest_name if dest_name else path.name
        with open(path, "rb") as f:
            # The server saves the file under prism_files_user/<upload_name>.
            resp = requests.post(
                f"{self._url}/files/prism",
                files={"file": (upload_name, f, "text/plain")},
                timeout=30,
            )
        resp.raise_for_status()
        # Returns e.g. "prism_files_user/dummy.prism" — pass directly to mc.cmd().
        return resp.json()["path"]

    def upload_prism_bundle(self, *paths: str) -> list[str]:
        """Upload a .prism file together with companion directories/files as a zip.

        Pass any mix of file and directory paths. All are zipped in memory and
        extracted into prism_files_user/ on the server, preserving structure.
        Use this for environments that ship a subfolder alongside the .prism file
        (e.g. icu_sepsis.prism + icu_sepsis/).

        Returns a list of "prism_files_user/<name>" paths for use in mc.cmd().

        Example::

            paths = mc.upload_prism_bundle("icu_sepsis.prism", "icu_sepsis/")
            job = mc.cmd(
                prism_file_path="/workspaces/coolmc/" + paths[0], ...)
        """
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in paths:
                path = Path(p)
                if path.is_dir():
                    # Add all files in the directory, preserving the folder name.
                    for f in path.rglob("*"):
                        if f.is_file():
                            zf.write(f, f.relative_to(path.parent))
                else:
                    zf.write(path, path.name)
        buf.seek(0)

        resp = requests.post(
            f"{self._url}/files/prism-bundle",
            files={"file": ("bundle.zip", buf, "application/zip")},
            timeout=120,  # large bundles (e.g. icu_sepsis) may take a while
        )
        resp.raise_for_status()
        return resp.json()["prism_files"]

    def list_prism_files(self) -> list[str]:
        """List all user-uploaded PRISM files available in the container."""
        return self._get("/files/prism")

    def delete_prism_file(self, filename: str) -> None:
        """Delete a previously uploaded PRISM file by filename."""
        resp = requests.delete(f"{self._url}/files/prism/{filename}", timeout=10)
        resp.raise_for_status()

    def last_run_id(self) -> str:
        """Return the mlflow_run_id of the most recently completed job."""
        job = Job._from_dict(self._get("/jobs/last"))
        return job.mlflow_run_id

    def _resolve_last_run_id(self) -> str:
        try:
            return self.last_run_id()
        except Exception as e:
            raise RuntimeError(
                'parent_run_id="last" requested but no completed jobs found. '
                "Run a training job first."
            ) from e

    def _post(self, path: str, payload: dict) -> dict:
        resp = requests.post(f"{self._url}{path}", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _get(self, path: str) -> Any:
        resp = requests.get(f"{self._url}{path}", timeout=10)
        resp.raise_for_status()
        return resp.json()
