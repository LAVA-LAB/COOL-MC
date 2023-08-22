import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import mlflow
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from common.rl_agents.agent import StochasticAgent
from common.utilities.helper import *

class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2, lr=0.001):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, a_size)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def get_action_idizes_not_zero(self, probs, prob_threshold):
        action_indexes = []
        for i, prob in enumerate(probs):
            if prob > prob_threshold:
                action_indexes.append(i)
        return action_indexes


    def act(self, state):
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action), int(torch.argmax(probs))

    def stochastic_act(self, state, prob_threshold):
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        probs = self.forward(state).cpu()
        return self.get_action_idizes_not_zero(probs.tolist()[0], prob_threshold), probs[0]

    def save_checkpoint(self, file_name : str):
        """Save model.

        Args:
            file_name (str): File name
        """
        torch.save(self.state_dict(), file_name)

    def load_checkpoint(self, file_name : str):
        """Load model

        Args:
            file_name (str): File name
        """
        self.load_state_dict(torch.load(file_name))

class ReinforceAgent(StochasticAgent):

    def __init__(self, state_dimension, hidden_layer_size, number_of_actions, gamma, lr):
        super().__init__()
        self.policy = Policy(state_dimension, hidden_layer_size, number_of_actions, lr)
        self.gamma = gamma
        self.rewards = []
        self.saved_log_probs = []


    def select_action(self, state : np.ndarray, deploy : bool =False):
        action_index, log_prob, max_action = self.policy.act(state)
        self.saved_log_probs.append(log_prob)
        return action_index



    def model_checking_select_action(self, state : np.ndarray, prob_threshold):
        return self.policy.stochastic_act(state, prob_threshold)

    def get_action_name_probability(self, env, action, state):
        # Action idx
        action_idx = env.action_mapper.action_name_to_action_index(action)
        # Action probability
        state = torch.from_numpy(np.array([state])).float().unsqueeze(0).to(DEVICE)
        probs = self.policy.forward(state).cpu()
        # probs to float
        print(probs.tolist())
        try:
            probs = probs.tolist()[0][0]
            return probs[action_idx]
        except:
            probs = self.policy.forward(state).cpu()
            robs = probs.tolist()[0]
            return probs[action_idx]







    def store_experience(self, state  : np.ndarray, action : int, reward : float, next_state : np.ndarray, terminal : bool):
        self.rewards.append(reward)


    def episodic_learn(self):
        discounts = [self.gamma**i for i in range(len(self.rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, self.rewards)])

        policy_loss = []
        for log_prob in self.saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        self.policy.optimizer.zero_grad()
        policy_loss.backward()
        self.policy.optimizer.step()

        self.rewards = []
        self.saved_log_probs = []



    def save(self, artifact_path='model'):
        """
        Saves the agent onto the MLFLow Server.
        """
        try:
            os.mkdir('tmp_model')
        except Exception as msg:
            pass
        self.policy.save_checkpoint('tmp_model/policy.chkpt')
        mlflow.log_artifacts("tmp_model", artifact_path=artifact_path)
        shutil.rmtree('tmp_model')

    def load(self, model_root_folder_path :str):
        """Loads the Agent from the MLFlow server.

        Args:
            model_root_folder_path (str): Model root folder path.
        """
        try:
            self.policy.load_checkpoint(os.path.join(model_root_folder_path,'policy.chkpt'))
        except:
            pass
