from sklearn.metrics import accuracy_score
import torch.nn.utils.prune as prune
import numpy as np
import torch
from common.interpreter.interpreter import *

def get_original_weights(model,layer_index):
    # Save a copy of original weights before pruning for all layers
    original_weights = model.layers[layer_index].weight.detach().clone()
    return original_weights
    
# Partitioned Permutation Importance for feature importance analysis
# For each feature assignment range of feature f, we create a feature importance vector
class FeaturePruner(Interpreter):

    def __init__(self, config):
        self.config = config
        print(self.config)
        self.name = self.config.split(";")[0]
        self.percentage = float(self.config.split(";")[1])
        try:
            self.file_name = self.config.split(";")[2]
            self.feature_names = self.config.split(";")[3].split(",")
        except:
            self.file_name = ""
            self.feature_names = self.config.split(";")[2].split(",")

    def feature_names_to_indices(self, env, feature_names):
        feature_indices = []
        for feature_name in feature_names:
            feature_indices.append(env.storm_bridge.state_mapper.mapper[feature_name])
        return feature_indices



    def interpret(self, env, m_project, model_checking_info):
        original_weights = get_original_weights(m_project.agent.q_eval,0)
        # Get indices
        neuron_indices = self.feature_names_to_indices(env, self.feature_names)
        # Create a binary mask of the same size as the layer weights
        mask = torch.ones_like(original_weights)
        # Set all outgoing connections from the neuron at position i to be pruned
        for neuron_index in neuron_indices:
            mask[:, neuron_index] = 0

        prune.custom_from_mask(m_project.agent.q_eval.layers[0], name="weight", mask=mask)
        prune.remove(m_project.agent.q_eval.layers[0], name="weight")

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
            f.write(f'{self.name},{model_checking_info["property"]},{self.percentage},{0},{mdp_reward_result}\n')
            f.close()
        #print("Run ID: " + run_id)
       