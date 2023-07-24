from sklearn.metrics import accuracy_score
import torch.nn.utils.prune as prune
import numpy as np
import torch
from common.interpreter.visualizer import *
from common.interpreter.interpreter import *

# Partitioned Permutation Importance for feature importance analysis
# For each feature assignment range of feature f, we create a feature importance vector
class RandomUnstructuredPruner(Interpreter):

    def __init__(self, config):
        self.config = config
        self.name = self.config.split(";")[0]
        self.percentage = float(self.config.split(";")[1])
        self.layer_index = int(self.config.split(";")[3])
        try:
            self.file_name = self.config.split(";")[2]
        except:
            self.file_name = ""


    def interpret(self, env, m_project, model_checking_info):
        original_weights = get_original_weights(m_project.agent.q_eval)
        # 1. For a percentage interval, plot all the safety measurements
        # 2. Plot the neural network with the highest percentage but the same safety measurement as the initial
        prune.random_unstructured(m_project.agent.q_eval.layers[self.layer_index], name="weight", amount=self.percentage)
        prune.remove(m_project.agent.q_eval.layers[self.layer_index], name="weight")
        mdp_reward_result, model_checking_info = env.storm_bridge.model_checker.induced_markov_chain(m_project.agent, m_project.preprocessors, env, m_project.command_line_arguments['constant_definitions'], m_project.command_line_arguments['prop'], False)
        #m_project.mlflow_bridge.log_result(mdp_reward_result)
        print("===============================")
        #run_id = m_project.mlflow_bridge.get_run_id()
        print(f'{model_checking_info["property"]}:\t{mdp_reward_result}')
        print(f'Model Size:\t\t{model_checking_info["model_size"]}')
        print(f'Number of Transitions:\t{model_checking_info["model_transitions"]}')
        print(f'Model Building Time:\t{model_checking_info["model_building_time"]}')
        print(f'Model Checking Time:\t{model_checking_info["model_checking_time"]}')
        print("Constant definitions:\t" + m_project.command_line_arguments['constant_definitions'])
        if self.file_name != "":
            f = open(self.file_name, "a")
            f.write(f'{self.name},{model_checking_info["property"]},{self.percentage},{self.layer_index},{mdp_reward_result}\n')
            f.close()
        #print("Run ID: " + run_id)
        #difference_plotting(m_project.agent.q_eval, original_weights)

