from common.interpreter.interpreter import Interpreter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

class ActionNameIgnorerInterpreter(Interpreter):

    def __init__(self, config):
        self.config = config

    def interpret(self, env, m_project, model_checking_info):
        all_results = []
        number_of_actions = env.action_mapper.get_action_count()
        action_names = []
        for action_index in range(0,number_of_actions):
            action_names.append(env.action_mapper.action_index_to_action_name(action_index))
        original_mdp_result = model_checking_info['mdp_reward_result']
        all_results.append(original_mdp_result)

        for action_index in range(1,number_of_actions):
            m_project.agent.block_nth_action = action_index
            mdp_reward_result, model_checking_info = env.storm_bridge.model_checker.induced_markov_chain(m_project.agent, m_project.preprocessors, env, m_project.command_line_arguments['constant_definitions'], m_project.command_line_arguments['prop'], False)
            all_results.append(mdp_reward_result)
        
        print("Property Query:", m_project.command_line_arguments['prop'])
        for i, result in enumerate(all_results):
            print(i, action_names[i], result)