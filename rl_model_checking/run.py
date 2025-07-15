import os
import sys
sys.path.insert(0, '../')
from common.utilities.helper import *
from common.utilities.training import *
from typing import Any, Dict
from common.utilities.project import Project
from common.safe_gym.safe_gym import SafeGym
from common.interpreter.interpreter_builder import *

def prepare_prop(prop):
    prepared = False
    original_prop = prop
    if prop.find("max") == -1 and prop.find("min") == -1:
        query = prop
        # Insert min at second position
        operator_str = query[:1]
        min_part = "min"
        prop = operator_str + min_part + query[1:]
        prepared = True
    return prop, prepared, original_prop




if __name__ == '__main__':
    os.environ['MKL_NUM_THREADS'] = '1'
    # Get command line arguments
    command_line_arguments = get_arguments()
    # Set seed
    set_random_seed(command_line_arguments['seed'])
    # Command line arguments set up
    command_line_arguments['task'] = RL_MODEL_CHECKING_TASK
    collect_label_and_states = (command_line_arguments['interpreter']!= '')
    # Get full prism_file_path
    prism_file_path = os.path.join(
        command_line_arguments['prism_dir'], command_line_arguments['prism_file_path'])

    # Load last run, if specified
    if command_line_arguments['parent_run_id'] == "last":
        command_line_arguments['project_name'], command_line_arguments['parent_run_id'] = LastRunManager.read_last_run()
        print(f"Loaded last run {command_line_arguments['parent_run_id']} from project {command_line_arguments['project_name']}")

    # Project
    m_project = Project(command_line_arguments)
    m_project.init_mlflow_bridge(
        command_line_arguments['project_name'], command_line_arguments['task'],
        command_line_arguments['parent_run_id'])
    m_project.load_saved_command_line_arguments()
    print("We use the following command line arguments:")
    print(m_project.command_line_arguments)

    # Project Environment
    prism_file_path = os.path.join(
        m_project.command_line_arguments['prism_dir'], m_project.command_line_arguments['prism_file_path'])
    env = SafeGym(prism_file_path, m_project.command_line_arguments['constant_definitions'],
                10, 1,
                  True,
                  m_project.command_line_arguments['seed'],
                  m_project.command_line_arguments['disabled_features'])

    # Create rest of project
    m_project.create_agent(m_project.command_line_arguments,
                           env.observation_space, env.action_space, env.action_mapper.actions)
    m_project.agent.load_env(env)
    m_project.create_preprocessor(m_project.command_line_arguments, env.observation_space, env.action_space, env.storm_bridge.state_mapper)
    m_project.create_postprocessor(m_project.command_line_arguments, env.observation_space, env.action_space, env.storm_bridge.state_mapper)

    # Prepare property
    m_project.command_line_arguments['prop'], prepared, original_prop = prepare_prop(m_project.command_line_arguments['prop'])

    # Set property as run name
    m_project.mlflow_bridge.set_property_query_as_run_name(original_prop + " for " + command_line_arguments['constant_definitions'])

    # Model checking
    mdp_reward_result, model_checking_info = env.storm_bridge.model_checker.induced_markov_chain(m_project.agent, m_project.preprocessors, env, m_project.command_line_arguments['constant_definitions'], m_project.command_line_arguments['prop'], collect_label_and_states)
    m_project.mlflow_bridge.log_result(mdp_reward_result)
    m_project.mlflow_bridge.log_number_of_states(model_checking_info['model_size'])
    m_project.mlflow_bridge.log_number_of_transitions(model_checking_info['model_transitions'])
    m_project.mlflow_bridge.log_faulty_outputs(model_checking_info['number_of_faulty_outputs'])
    m_project.mlflow_bridge.log_time_in_seconds(model_checking_info['model_building_time'] + model_checking_info['model_checking_time'])

    run_id = m_project.mlflow_bridge.get_run_id()
    print(f'{original_prop}:\t{mdp_reward_result}')
    print(f'Model Size:\t\t{model_checking_info["model_size"]}')
    print(f'Number of Transitions:\t{model_checking_info["model_transitions"]}')
    print(f'Model Building Time:\t{model_checking_info["model_building_time"]}')
    print(f'Model Checking Time:\t{model_checking_info["model_checking_time"]}')
    print("Constant definitions:\t" + m_project.command_line_arguments['constant_definitions'])
    print("Run ID: " + run_id)

    # Interpreter
    interpreter = InterpreterBuilder.build_interpreter(m_project.command_line_arguments['interpreter'])
    if interpreter != None:
        interpreter.interpret(env, m_project.agent, model_checking_info)


    m_project.save()
    m_project.close()
