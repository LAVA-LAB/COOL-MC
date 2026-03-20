"""COOL-MC Python client.

Importing this package automatically starts the COOL-MC Docker container
(if it is not already running) before returning control to the caller.

Usage::

    import coolmc

    mc = coolmc.CoolMC()
    job = mc.cmd(
        task="safe_training",
        prism_file_path="frozen_lake.prism",
        algorithm="dqn_agent",
        num_episodes=500,
        project_name="my_experiment",
    )
    job = mc.wait(job.job_id)
    print(job.mlflow_run_id)
    print(mc.get_logs(job.job_id))
"""

from .docker_manager import ensure_running
from .client import CoolMC, Job

ensure_running()

__all__ = ["CoolMC", "Job"]
