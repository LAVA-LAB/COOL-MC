"""Configuration for the COOL-MC client."""

from pathlib import Path

SERVER_HOST = "localhost"
SERVER_PORT = 8765
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"

CONTAINER_NAME = "coolmc-server"
IMAGE_NAME = "coolmc-server"
WORKDIR = "/workspaces/coolmc"

# All persistent data lives here on the host — survives pip upgrades and
# container rebuilds. Created automatically on first import.
COOLMC_HOME = Path.home() / ".coolmc"
VOLUMES_DIR = COOLMC_HOME / "volumes"

# Defaults mirrored from common/utilities/helper.py
DEFAULTS = {
    # Meta
    "task": "safe_training",
    "project_name": "defaultproject",
    "parent_run_id": "",
    "prism_dir": "../prism_files",
    "prism_file_path": "transporter.prism",
    "constant_definitions": "",
    "disabled_features": "",
    "seed": -1,
    "training_threshold": -1000000000000,
    # Training
    "num_episodes": 1000,
    "eval_interval": 9,
    "sliding_window_size": 100,
    "reward_flag": 0,
    "max_steps": 100,
    "wrong_action_penalty": 1000,
    "deploy": 0,
    # Behavioral Cloning
    "bc_epochs": 100,
    "accuracy_threshold": 100.0,
    "behavioral_cloning": "",
    # Preprocessor / Postprocessor / Interpreter / State Labeler
    "preprocessor": "",
    "postprocessor": "",
    "interpreter": "",
    "state_labeler": "",
    # Model Checking
    "prop": "",
    "range_plotting": 1,
    # Agent
    "algorithm": "dqn_agent",
    "alpha": 0.99,
    "noise_scale": 1e-2,
    "layers": 2,
    "neurons": 64,
    "replay_buffer_size": 300000,
    "epsilon": 1,
    "epsilon_dec": 0.9999,
    "epsilon_min": 0.1,
    "gamma": 0.99,
    "replace": 304,
    "lr": 0.0001,
    "batch_size": 32,
}
