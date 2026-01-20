import mlflow
import os
import shutil
from typing import List
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from collections import deque
import math
import torch
import numpy as np
from common.agents.agent import Agent
from common.agents.dqn_agent import DQNAgent
from common.agents.partial_observable_manager import *




class TurnBasedNAgents(Agent):

    def __init__(self, command_line_arguments, state_dimension : int, number_of_actions : int, number_of_neurons : int):
        """Initialize Deep Q-Learning Agent

        Args:
            state_dimension (np.array): State Dimension
            number_of_actions (int): Number of Actions
        """
        self.number_of_actions = number_of_actions
        self.state_dimension = state_dimension
        self.number_of_neurons = number_of_neurons
        self.command_line_arguments = command_line_arguments
        self.number_of_agents = self.extract_number_of_players(os.path.join(command_line_arguments['prism_dir'], command_line_arguments['prism_file_path']))

    def extract_number_of_players(self, prism_file_path):
        """
        Extracts the number of players from the prism file.
        """
        file1 = open(prism_file_path, 'r')
        lines = file1.readlines()
        for line in lines:
            if "const int NUMBER_OF_PLAYERS" in line:
                number = line.split("=")[1]
                number = int(number.split(";")[0].strip())+1
                print("Number of Agents: ", number)
                return number


    def load_env(self, env):
        """
        Load the environment.
        """
        try:
            self.turn_idx = env.storm_bridge.state_mapper.mapper["turn"]
        except Exception as e:
            print(e)




    def save(self):
        """
        Saves the agent onto the MLFLow Server.
        """
        for i in range(len(self.agents)):
            self.agents[i].save(artifact_path='model'+str(i))
            self.agents[i].save(artifact_path='model'+str(i))



    def load(self, model_root_folder_path):
        """Loads the Agent from the MLFlow server.

        Args:
            model_root_folder_path (str): Model root folder path.
        """
        self.agents = []
        self.model_root_folder_path = model_root_folder_path
        #print(self.model_root_folder_path)
        for i in range(self.number_of_agents):
            # Extract state_dimension from prism file for each agent
            self.agents.append(DQNAgent(self.state_dimension, self.number_of_neurons, self.number_of_actions, epsilon=self.command_line_arguments['epsilon'], epsilon_dec=self.command_line_arguments['epsilon_dec'], epsilon_min=self.command_line_arguments['epsilon_min'], gamma=self.command_line_arguments['gamma'], learning_rate=self.command_line_arguments['lr'], replace=self.command_line_arguments['replace'], batch_size=self.command_line_arguments['batch_size'], replay_buffer_size=self.command_line_arguments['replay_buffer_size']))
            self.agents[i].load(self.model_root_folder_path+str(i))




    def store_experience(self, state : np.array, action : int, reward : float, n_state : np.array, done : bool):
        """Stores experience into the replay buffer

        Args:
            state (np.array): State
            action (int): action
            reward (float): reward
            n_state (np.array): next state
            done (bool): Terminal state?
        """
        self.turn_value = int(state[self.turn_idx])
        self.agents[self.turn_value].store_experience(state, action, reward, n_state, done)




    def select_action(self, state : np.ndarray, deploy=False, attack=None) -> int:
        """Select random action or action based on the current state.

        Args:
            state (np.ndarray): Current state
            deploy (bool, optional): If true, no random states. Defaults to False.

        Returns:
            [type]: action
        """
        self.turn_value = int(state[self.turn_idx])
        action_idx = self.agents[self.turn_value].select_action(state, deploy)
        return action_idx



    def step_learn(self):
        """
        Agent learning.
        """
        self.agents[self.turn_value].step_learn()



