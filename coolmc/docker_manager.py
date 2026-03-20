"""Manages the COOL-MC Docker container lifecycle."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import requests

from .config import CONTAINER_NAME, SERVER_URL, VOLUMES_DIR, WORKDIR

# The static compose file ships with the package (used only for the build
# context and image definition). At runtime we write a second, resolved
# compose file to ~/.coolmc/ so that volume paths are absolute and correct
# regardless of where pip installed the package.
_PACKAGE_DIR = Path(__file__).parent        # coolmc/ inside site-packages
_RUNTIME_COMPOSE = VOLUMES_DIR.parent / "docker-compose.yml"  # ~/.coolmc/docker-compose.yml

_HEALTHZ = f"{SERVER_URL}/healthz"
_STARTUP_TIMEOUT = 120  # seconds


def ensure_running() -> None:
    """Start the container if not already running, then wait for readiness."""
    if _is_healthy():
        return

    _prepare_volumes()
    _write_runtime_compose()
    _docker_compose_up()
    _wait_for_healthy()


def _prepare_volumes() -> None:
    """Create persistent volume directories and seed jobs.json if missing."""
    for subdir in ("mlruns", "logs", "prism_files"):
        (VOLUMES_DIR / subdir).mkdir(parents=True, exist_ok=True)
    jobs_file = VOLUMES_DIR / "jobs.json"
    if not jobs_file.exists():
        jobs_file.write_text("{}")
    _init_mlflow_store()


def _init_mlflow_store() -> None:
    """Create the MLflow default experiment (ID 0) directory if it doesn't exist.

    cool_mc.py calls mlflow.run() which hardcodes experiment_id=0. MLflow's
    file store requires mlruns/0/meta.yaml to exist before any run is created.
    """
    exp0_dir = VOLUMES_DIR / "mlruns" / "0"
    meta = exp0_dir / "meta.yaml"
    if meta.exists():
        return
    exp0_dir.mkdir(parents=True, exist_ok=True)
    # artifact_location must match the path inside the container
    meta.write_text(
        "artifact_location: file:///workspaces/coolmc/mlruns/0\n"
        "experiment_id: '0'\n"
        "lifecycle_stage: active\n"
        "name: Default\n"
    )


def _write_runtime_compose() -> None:
    """Write a docker-compose.yml with fully resolved absolute volume paths.

    The static compose file shipped in the package uses relative paths that
    only work when docker compose is run from the repo root. After a pip
    install the package is in site-packages, so we generate a resolved copy
    at ~/.coolmc/docker-compose.yml instead.
    """
    mlruns   = VOLUMES_DIR / "mlruns"
    logs     = VOLUMES_DIR / "logs"
    prism    = VOLUMES_DIR / "prism_files"
    jobs     = VOLUMES_DIR / "jobs.json"

    content = f"""\
services:
  coolmc-server:
    build:
      context: {_PACKAGE_DIR}
      dockerfile: Dockerfile
    container_name: {CONTAINER_NAME}
    working_dir: {WORKDIR}
    ports:
      - "8765:8765"
      - "5000:5000"
    restart: unless-stopped
    volumes:
      - {mlruns}:{WORKDIR}/mlruns
      - {logs}:{WORKDIR}/logs
      - {prism}:{WORKDIR}/prism_files_user
      - {jobs}:{WORKDIR}/jobs.json
"""
    _RUNTIME_COMPOSE.parent.mkdir(parents=True, exist_ok=True)
    _RUNTIME_COMPOSE.write_text(content)


def _is_healthy() -> bool:
    try:
        r = requests.get(_HEALTHZ, timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def stop() -> None:
    """Stop the COOL-MC Docker container."""
    subprocess.run(
        ["docker", "compose", "-f", str(_RUNTIME_COMPOSE), "down"],
        check=True,
    )


def restart() -> None:
    """Stop the container, rebuild if needed, and wait for it to be healthy."""
    stop()
    _docker_compose_up()
    _wait_for_healthy()


def _docker_compose_up() -> None:
    subprocess.run(
        ["docker", "compose", "-f", str(_RUNTIME_COMPOSE), "up", "-d", "--build"],
        check=True,
    )


def _wait_for_healthy() -> None:
    deadline = time.time() + _STARTUP_TIMEOUT
    while time.time() < deadline:
        if _is_healthy():
            return
        time.sleep(2)
    raise RuntimeError(
        f"COOL-MC container did not become healthy within {_STARTUP_TIMEOUT}s. "
        f"Check: docker logs {CONTAINER_NAME}"
    )
