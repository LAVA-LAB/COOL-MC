from common.interpreter.interpreter import Interpreter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

class NthActionInterpreter(Interpreter):

    def __init__(self, config):
        self.config = config

    def interpret(self, env, m_project, model_checking_info):
        all_results = []
        number_of_actions = env.action_mapper.get_action_count()
        original_mdp_result = model_checking_info['mdp_reward_result']
        all_results.append(original_mdp_result)

        for action_index in range(1,number_of_actions):
            drn_file_name = "test_" + str(action_index) + ".drn"
            m_project.agent.nth_action_instead = action_index
            mdp_reward_result, model_checking_info = env.storm_bridge.model_checker.induced_markov_chain(m_project.agent, m_project.preprocessors, env, m_project.command_line_arguments['constant_definitions'], m_project.command_line_arguments['prop'], False, drn_file_name)
            all_results.append(mdp_reward_result)
        
        print("Property Query:", m_project.command_line_arguments['prop'])
        for i, result in enumerate(all_results):
            print(i, result)