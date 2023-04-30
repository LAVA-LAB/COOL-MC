

import mlflow
import os
import shutil
from typing import List
from common.rl_agents.agent import DeterministicAgent
import numpy as np
import numpy as np
import os

class HillClimbingAgent(DeterministicAgent):

    def __init__(self, state_dimension, number_of_actions, gamma, noise_scale):
        super().__init__()
        self.state_dimension = state_dimension
        self.number_of_actions = number_of_actions
        self.w =  1e-4*np.random.rand(state_dimension, number_of_actions)
        self.rewards = []
        self.gamma = gamma
        self.best_R = -np.inf
        self.best_w = np.random.rand(state_dimension, number_of_actions)
        self.noise_scale = noise_scale
        self.progress_counter = 0


    def select_action(self, state : np.ndarray, deploy : bool =False):
        x = None
        if deploy:
            x = np.dot(state, self.best_w)
        else:
            x = np.dot(state, self.w)
        probs = np.exp(x)/sum(np.exp(x))
        action_index = np.argmax(probs)
        return int(action_index)


    def store_experience(self, state  : np.ndarray, action : int, reward : float, next_state : np.ndarray, terminal : bool):
        reward = float(reward)
        self.rewards.append(reward)


    def episodic_learn(self):
        discounts = [self.gamma**i for i in range(len(self.rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, self.rewards)])
        if R >= self.best_R: # found better weights
            self.best_R = R
            self.best_w = self.w
            self.noise_scale = max(1e-3, self.noise_scale / 2)
            self.w += self.noise_scale * np.random.rand(*self.w.shape)
        else: # did not find better weights
            self.noise_scale = min(2, self.noise_scale * 2)
            self.w = self.best_w + self.noise_scale * np.random.rand(*self.w.shape)
            self.progress_counter += 1
            if self.progress_counter >= 10000:
                self.w = self.noise_scale * np.random.rand(*self.w.shape)

        self.rewards = []


    def save(self):
        try:
            os.mkdir('tmp_model')
        except Exception as msg:
            pass
        np.save(os.path.join("tmp_model", 'model.npy'), self.best_w)
        mlflow.log_artifacts("tmp_model", artifact_path="model")
        shutil.rmtree('tmp_model')


    def load(self, root_folder):
        try:
            self.best_w = np.load(os.path.join(root_folder, 'model.npy'))
            self.w = np.load(os.path.join(root_folder, 'model.npy'))
        except:
            self.best_w = np.random.rand(self.state_dimension, self.number_of_actions)
            self.w = np.random.rand(self.state_dimension, self.number_of_actions)
