"""This module provides helper functions for COOL-MC."""
import argparse
import sys
import random
from typing import Any, Dict
import numpy as np
import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SAFE_TRAINING_TASK = "safe_training"
RL_MODEL_CHECKING_TASK = "rl_model_checking"
DEFAULT_TRAINING_THRESHOLD = -1000000000000

def get_arguments() -> Dict[str, Any]:
    """Parses all the COOL-MC arguments
    Returns:
        Dict[str, Any]: dictionary with the command line arguments as key and their assignment as value
    """
    arg_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args = argparse.Namespace()

    # Meta
    arg_parser.add_argument('--task', help=f'What type of task do you want to perform({SAFE_TRAINING_TASK}, rl_model_checking)?', type=str,
                            default=SAFE_TRAINING_TASK)
    arg_parser.add_argument('--project_name', help='What is the name of your project?', type=str,
                            default='defaultproject')
    arg_parser.add_argument('--parent_run_id', help='Do you want to continue training of a RL agent? Name the run_id of the last training unit (see mlflow ui).', type=str,
                            default='')
    arg_parser.add_argument('--prism_dir', help='In which folder should we look for the PRISM environment?', type=str,
                            default='../prism_files')
    arg_parser.add_argument('--prism_file_path', help='Whats the name of the prism file?', type=str,
                            default='transporter.prism')
    arg_parser.add_argument('--constant_definitions', help='Constant definitions of the formal model (PRISM model)', type=str,
                            default='')
    arg_parser.add_argument('--disabled_features', help='Features which should not be used by the RL agent: FEATURE1,FEATURE2,...', type=str,
                            default='')
    arg_parser.add_argument('--seed', help='Random Seed for numpy, random, storm, pytorch', type=int,
                            default=-1)
    arg_parser.add_argument('--training_threshold', help='Range Plotting Flag.', type=float,
                            default=DEFAULT_TRAINING_THRESHOLD)
    # Training
    arg_parser.add_argument('--num_episodes', help='What is the number of training episodes?', type=int,
                            default=1000)
    arg_parser.add_argument('--eval_interval', help='Monitor every eval_interval episodes.', type=int,
                            default=9)
    arg_parser.add_argument('--sliding_window_size', help='What is the sliding window size for the reward averaging?', type=int,
                            default=100)
    arg_parser.add_argument('--reward_flag', help='Reward Flag (0=penalty,1=reward).', type=int,
                            default=0)
    arg_parser.add_argument('--max_steps', help='Maximal steps in environment.', type=int,
                            default=100)
    arg_parser.add_argument('--wrong_action_penalty', help='Wrong action penalty.', type=int,
                            default=1000)
    arg_parser.add_argument('--deploy', help='Deploy Flag (0=no deploy, 1=deploy).', type=int,
                            default=0)

    # Behavioral Cloning
    arg_parser.add_argument('--bc_epochs', help='What is the number of training epochs for behavioral cloning?', type=int,
                            default=100)
    arg_parser.add_argument('--behavioral_cloning', help='Preprocessor configuration string.', type=str,
                            default='')

    # Preprocessor
    arg_parser.add_argument('--preprocessor', help='Preprocessor configuration string.', type=str,
                            default='')

    # Manipulator
    arg_parser.add_argument('--postprocessor', help='Manipulator configuration string.', type=str,
                            default='')

    # Interpreter
    arg_parser.add_argument('--interpreter', help='Interpreter configuration string.', type=str,
                            default='')

    # State Labeler
    arg_parser.add_argument('--state_labeler', help='State labeler configuration string (e.g., "critical_state;min=0.3;max=0.7").', type=str,
                            default='')

    # Model Checking
    arg_parser.add_argument('--prop', help='Property Specification.', type=str,
                            default='')
    arg_parser.add_argument('--range_plotting', help='Range Plotting Flag.', type=int,
                            default=1)

    # Agents
    arg_parser.add_argument('--algorithm', help='What is the used agent algorithm?', type=str,
                            default='dqn_agent')
    arg_parser.add_argument('--alpha', help='Alpha', type=float,
                            default=0.99)
    arg_parser.add_argument('--noise_scale', help='Noise Scale for Hillclimbing', type=float,
                            default=1e-2)
    arg_parser.add_argument('--layers', help='Number of layers', type=int,
                            default=2)
    arg_parser.add_argument('--neurons', help='Number of neurons per layer', type=int,
                            default=64)
    arg_parser.add_argument('--replay_buffer_size', help='Replay buffer size', type=int,
                            default=300000)
    arg_parser.add_argument('--epsilon', help='Epsilon Starting Rate', type=float,
                            default=1)
    arg_parser.add_argument('--epsilon_dec', help='Epsilon Decreasing Rate', type=float,
                            default=0.9999)
    arg_parser.add_argument('--epsilon_min', help='Minimal Epsilon Value', type=float,
                            default=0.1)
    arg_parser.add_argument('--gamma', help='Gamma', type=float,
                            default=0.99)
    arg_parser.add_argument('--replace', help='Replace Target Network Intervals', type=int,
                            default=304)
    arg_parser.add_argument('--lr', help='Learning Rate', type=float,
                            default=0.0001)
    arg_parser.add_argument('--batch_size', help='Batch Size', type=int,
                            default=32)

    args, _ = arg_parser.parse_known_args(sys.argv)
    return vars(args)

def parse_prop_type(prop: str) -> str:
    """Guides in Safe training, if we want to maximize or minimize the property result
    or do normal reward maximization.
    Args:
        prop (str): Property Query
    Returns:
        str: type of optimization
    """
    assert isinstance(prop, str)
    if prop.find("min") < prop.find("=") and prop.find("min") != -1:
        return "min_prop"
    if prop.find("max") < prop.find("=") and prop.find("max") != -1:
        return "max_prop"
    return "reward"


def set_random_seed(seed: int):
    """Set global seed to all used libraries. If you use other libraries too,
    add them here.
    Args:
        seed (int): Random Seed
    """
    assert isinstance(seed, int)
    if seed != -1:
        print("Set Seed to", seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class LastRunManager:

    @staticmethod
    def write_last_run(project_name, run_id):
        # Write last run to file
        with open("../last_run.txt", "w") as f:
            f.write(project_name + "," + str(run_id))

    @staticmethod
    def read_last_run():
        # Read last run from file
        with open("../last_run.txt", "r") as f:
            project_name, run_id = f.read().split(",")
            return project_name, run_id
