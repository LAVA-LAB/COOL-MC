import mlflow
import os
import shutil
from typing import List
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.rl_agents.agent import Agent
from collections import OrderedDict
import torch
import numpy as np
from common.rl_agents.dqn_agent import DQNAgent
from common.utilities.helper import *
import numpy as np

class PartialObservableManager:

    def __init__(self, prism_file_path, state_mapper) -> None:
        self.prism_file_path = prism_file_path
        # Read text file
        with open(self.prism_file_path, 'r') as file:
            self.prism_lines = file.readlines()
        self.state_mapper = state_mapper
        self.all_agent_variables = self.extract_partial_observable_state_variables(self.prism_lines)

    def extract_partial_observable_state_variables(self, path):
        """Extract the partial observable state variables from the PRISM file.

        Args:
            path (str): Path to the PRISM file

        Returns:
            list: List of partial observable state variables
        """
        assert isinstance(path, list)
        all_agent_variables = []
        agent_counter=0
        for i, line in enumerate(path):
            if line.startswith('//AGENT'):
                agent_variables = []
                line_parts = line.split(":")
                variable_parts = line_parts[1].split(" ")
                for j, variable_part in enumerate(variable_parts):
                    if variable_part.strip() != '':
                        agent_variables.append(variable_part.strip())
                agent_counter+=1
                all_agent_variables.append(agent_variables)
        return all_agent_variables

    def get_observation_dimension_for_agent_idx(self, idx):
        return len(self.all_agent_variables[idx])

    def get_observation(self, full_state, agent_index):
        observation = []
        for variable_name in self.all_agent_variables[agent_index]:
            if variable_name.strip() != '':
                #print(self.state_mapper.mapper)
                variable_idx = self.state_mapper.mapper[variable_name.strip()]
                observation.append(full_state[variable_idx])
        return np.array(observation)

    def inject_observation_into_state(self, full_state, observation, agent_index):
        for i, variable_name in enumerate(self.all_agent_variables[agent_index]):
            if variable_name.strip() != '':
                print(variable_name.strip(), observation[i])
                full_state[self.state_mapper.mapper[variable_name.strip()]] = observation[i]
        return full_state



class CooperativePOAgents(Agent):

    def __init__(self, command_line_arguments, state_dimension : int, number_of_actions : int, combined_actions, number_of_neurons : int):
        """Initialize Deep Q-Learning Agent

        Args:
            state_dimension (np.array): State Dimension
            number_of_actions (int): Number of Actions
        """
        self.number_of_actions = number_of_actions
        self.state_dimension = state_dimension
        self.combined_actions = combined_actions
        self.combined_actions.sort()
        self.number_of_neurons = number_of_neurons
        # Extract all actions for each agent (for each list element, there is a list of actions for the corresponding agent at position i)
        self.all_actions = self.extract_actions_for_agents(self.combined_actions)
        self.command_line_arguments = command_line_arguments


    def load_env(self, env):
        """
        Load the environment.
        """
        print("LOAD ENVIRONMENT")
        self.env = env
        prism_path = os.path.join(self.command_line_arguments['prism_dir'], self.command_line_arguments['prism_file_path'])
        self.po_manager = PartialObservableManager(prism_path, self.env.storm_bridge.state_mapper)
        # Extract number of agents
        self.agents = []
        for i in range(len(self.all_actions)):
            # Extract state_dimension from prism file for each agent
            state_dimension = self.po_manager.get_observation_dimension_for_agent_idx(i)
            self.agents.append(DQNAgent(state_dimension, self.number_of_neurons,len(self.all_actions[i]), epsilon=self.command_line_arguments['epsilon'], epsilon_dec=self.command_line_arguments['epsilon_dec'], epsilon_min=self.command_line_arguments['epsilon_min'], gamma=self.command_line_arguments['gamma'], learning_rate=self.command_line_arguments['lr'], replace=self.command_line_arguments['replace'], batch_size=self.command_line_arguments['batch_size'], replay_buffer_size=self.command_line_arguments['replay_buffer_size']))
            #print(self.model_root_folder_path)
            self.agents[i].load(self.model_root_folder_path+str(i))



    def extract_actions_for_agents(self, actions):
        """
        Extract actions for each agent.
        """
        all_actions = []
        for combined_action in actions:
            # take1_open2_close3 (number indicate agent)
            actions = combined_action.split('_')
            all_actions.append(actions)
        # Each row i contains now all the actions of agent i
        all_actions = list(zip(*all_actions))
        for i in range(len(all_actions)):
            all_actions[i] = list(all_actions[i])
            all_actions[i] = list(set(all_actions[i]))
            # Sort for each agent the actions
            all_actions[i].sort()
        return all_actions


    def local_action_index_to_name(self, agent_idx, action_index):
        """
        Convert the internal action index to the name of the action.
        """
        return self.all_actions[agent_idx][action_index]

    def local_action_name_to_idx(self, agent_idx, action_name):
        return self.all_actions[agent_idx].index(action_name)


    def combine_actions(self, actions):
        """
        Combine actions into a single action.
        """
        combined_action = ''
        for idx, action_idx in enumerate(actions):
            combined_action += self.local_action_index_to_name(idx, actions[idx]) + '_'

        selected_actions = combined_action[:-1]
        return self.combined_actions.index(selected_actions)


    def save(self):
        """
        Saves the agent onto the MLFLow Server.
        """
        for i in range(len(self.agents)):
            self.agents[i].save(artifact_path='model'+str(i))


    def load(self, model_root_folder_path):
        """Loads the Agent from the MLFlow server.

        Args:
            model_root_folder_path (str): Model root folder path.
        """
        self.model_root_folder_path = model_root_folder_path
        # LOAD IN LOAD ENV



    def store_experience(self, state : np.array, action : int, reward : float, n_state : np.array, done : bool):
        """Stores experience into the replay buffer

        Args:
            state (np.array): State
            action (int): action
            reward (float): reward
            n_state (np.array): next state
            done (bool): Terminal state?
        """
        combined_action_name = self.combined_actions[action]
        for agent_idx, action_name in enumerate(combined_action_name.split("_")):
            observation = self.po_manager.get_observation(state, agent_idx)
            n_observation = self.po_manager.get_observation(n_state, agent_idx)
            local_action_idx = self.local_action_name_to_idx(agent_idx, action_name)
            self.agents[agent_idx].store_experience(observation, local_action_idx, reward, n_observation, done)



    def select_action(self, state : np.ndarray, deploy=False, attack=None) -> int:
        """Select random action or action based on the current state.

        Args:
            state (np.ndarray): Current state
            deploy (bool, optional): If true, no random states. Defaults to False.

        Returns:
            [type]: action
        """
        all_action_idizes = []
        for i in range(len(self.agents)):
            observation = self.po_manager.get_observation(state, i)
            action_idx = self.agents[i].select_action(observation, deploy)
            all_action_idizes.append(action_idx)

        action_idx = self.combine_actions(all_action_idizes)
        return action_idx



    def step_learn(self):
        """
        Agent learning.
        """
        for i in range(len(self.agents)):
            self.agents[i].step_learn()



