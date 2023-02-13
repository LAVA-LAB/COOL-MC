import os
import sys
sys.path.insert(0, '../')
from common.utilities.helper import *
from common.utilities.training import *
from typing import Any, Dict
from common.utilities.project import Project
#from common.utilities.training import train
#from common.safe_gym.safe_gym import SafeGym


if __name__ == '__main__':
    os.environ['MKL_NUM_THREADS'] = '1'
    command_line_arguments = get_arguments()
    set_random_seed(command_line_arguments['seed'])
    # Command line arguments set up
    command_line_arguments['task'] = SAFE_TRAINING_TASK
    command_line_arguments['prop_type'] = parse_prop_type(
        command_line_arguments['prop'])
    command_line_arguments['reward_flag'] = command_line_arguments['reward_flag'] == 1
    command_line_arguments['deploy'] = (1 == command_line_arguments['deploy'])
    prism_file_path = os.path.join(
        command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])

    if command_line_arguments['parent_run_id'] == "last":
        command_line_arguments['project_name'], command_line_arguments['parent_run_id'] = LastRunManager.read_last_run()
        print("Loaded last run: ", command_line_arguments['project_name'], command_line_arguments['parent_run_id'])
    # Environment
    env = SafeGym(prism_file_path, command_line_arguments['constant_definitions'],
                command_line_arguments['max_steps'], command_line_arguments['wrong_action_penalty'],
                  command_line_arguments['reward_flag'],
                  command_line_arguments['seed'],
                  command_line_arguments['disabled_features'])

    # Project
    m_project = Project(command_line_arguments)
    m_project.init_mlflow_bridge(
        command_line_arguments['project_name'], command_line_arguments['task'],
        command_line_arguments['parent_run_id'])
    m_project.load_saved_command_line_arguments()

    #print(m_project.command_line_arguments)
    m_project.create_agent(command_line_arguments,
                           env.observation_space, env.action_space, env.action_mapper.actions)
    m_project.create_preprocessor(command_line_arguments, env.observation_space, env.action_space, env.storm_bridge.state_mapper)
    m_project.create_postprocessor(command_line_arguments, env.observation_space, env.action_space, env.storm_bridge.state_mapper)
    m_project.mlflow_bridge.set_property_query_as_run_name(
        command_line_arguments['prop'] + " for " + command_line_arguments['constant_definitions'])

    # Train
    train(m_project, env, prop_type=command_line_arguments['prop_type'])
    run_id = m_project.mlflow_bridge.get_run_id()
    print("Run ID: " + run_id)
    LastRunManager.write_last_run(m_project.command_line_arguments['project_name'], run_id)
    m_project.close()
