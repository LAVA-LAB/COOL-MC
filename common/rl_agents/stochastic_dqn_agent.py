import mlflow
import os
import shutil
from typing import List
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.rl_agents.agent import StochasticAgent
from common.utilities.helper import *
from collections import OrderedDict
import torch
import numpy as np
from torch.distributions import Categorical


class ReplayBuffer(object):
    def __init__(self, max_size : int, state_dimension : int):
        """
        Initialize the replay buffer. The Replay Buffer stores the RL agent experiences.
        Each row is a transition and contains:
        a state, the action to the next state, the received reward, the next state, and if the environment is done.



        Args:
            max_size ([int]): The maximal rows of the replay buffer
            state_dimension ([int]): The shape of the state
        """
        self.size = max_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.size, state_dimension),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.size, state_dimension),
                                         dtype=np.float32)

        self.action_memory = np.zeros(self.size, dtype=np.int64)
        self.reward_memory = np.zeros(self.size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.size, dtype=np.bool)

    def store_transition(self, state : np.array, action : int, reward : float, state_ : np.array, done : bool):
        """Stores the transition into the replay buffer.

        Args:
            state (np.array): State
            action (int): Action
            reward (float): Reward
            state_ (np.array): Next State
            done (bool): episode done?
        """
        index = self.memory_counter % self.size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.memory_counter += 1

    def sample(self, batch_size : int) -> tuple:
        """Sample a batch for learning

        Args:
            batch_size (int): Batch Size

        Returns:
            tuple: A tuple of batches (states, actions, rewards, next state, terminal)
        """
        max_mem = min(self.memory_counter, self.size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return torch.tensor(states).to(DEVICE), torch.tensor(actions).to(DEVICE), torch.tensor(rewards).to(DEVICE), torch.tensor(states_).to(DEVICE), torch.tensor(terminal).to(DEVICE)

class DeepQNetwork(nn.Module):
    def __init__(self, state_dimension : int, number_of_neurons : List[int], number_of_actions : int, lr : float):
        """

        Args:
            state_dim (int): The dimension of the state
            number_of_neurons (List[int]): List of neurons. Each element is a new layer.
            number_of_actions (int): Number of actions.
            lr (float): Learning rate.
        """
        super(DeepQNetwork, self).__init__()

        layers = OrderedDict()
        previous_neurons = state_dimension
        for i in range(len(number_of_neurons)+1):
            if i == len(number_of_neurons):
                layers[str(i)] = torch.nn.Linear(previous_neurons, number_of_actions)
            else:
                layers[str(i)]  = torch.nn.Linear(previous_neurons, number_of_neurons[i])
                previous_neurons = number_of_neurons[i]
        self.layers = torch.nn.Sequential(layers)


        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.DEVICE = DEVICE
        self.to(self.DEVICE)


    def forward(self, state : np.array) -> int:
        """[summary]

        Args:
            state (np.array): State

        Returns:
            int: Action Index
        """
        try:
            x = state
            for i in range(len(self.layers)):
                if i == (len(self.layers)-1):
                    x = self.layers[i](x)
                    x = F.softmax(x, dim=0)  # applying softmax layer
                else:
                    x = F.relu(self.layers[i](x))
            return x
        except:
            state = torch.tensor(state).float().to(DEVICE)
            x = state
            for i in range(len(self.layers)):
                if i == (len(self.layers)-1):
                    x = self.layers[i](x)
                    x = F.softmax(x, dim=0)  # applying softmax layer
                else:
                    x = F.relu(self.layers[i](x))
            return x

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


class StochasticDQNAgent(StochasticAgent):

    def __init__(self, state_dimension : int, number_of_neurons, number_of_actions : List[int], gamma : float = 0.99, learning_rate : float = 0.001, replace : int =100, batch_size : int =64, replay_buffer_size : int =10000):
        """Initialize Deep Q-Learning Agent

        Args:
            state_dimension (np.array): State Dimension
            number_of_neurons (List[int]): Each element is a new layer and defines the number of neurons in the current layer
            number_of_actions (int): Number of Actions
            gamma (float, optional): Gamma value. Defaults to 0.99.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            replace (int, optional): Replace Interval of the target network (q_next). Defaults to 100.
            batch_size (int, optional): Batch Size. Defaults to 64.
            replay_buffer_size (int, optional): Replay Buffer size. Defaults to 10000.
        """
        self.number_of_actions = number_of_actions
        self.replay_buffer = ReplayBuffer(replay_buffer_size, state_dimension)
        self.q_eval = DeepQNetwork(state_dimension, number_of_neurons, number_of_actions, learning_rate)
        self.q_next = DeepQNetwork(state_dimension, number_of_neurons, number_of_actions, learning_rate)
        self.gamma = torch.tensor(gamma).to(DEVICE)
        self.replace = replace
        self.batch_size = batch_size
        self.exp_counter = 0
        self.learn_step_counter = 0

    def save(self, artifact_path='model'):
        """
        Saves the agent onto the MLFLow Server.
        """
        try:
            os.mkdir('tmp_model')
        except Exception as msg:
            import getpass
            username = getpass.getuser()
        self.q_eval.save_checkpoint('tmp_model/q_eval.chkpt')
        self.q_next.save_checkpoint('tmp_model/q_next.chkpt')
        mlflow.log_artifacts("tmp_model", artifact_path=artifact_path)
        shutil.rmtree('tmp_model')

    def load(self, model_root_folder_path :str):
        """Loads the Agent from the MLFlow server.

        Args:
            model_root_folder_path (str): Model root folder path.
        """
        try:
            self.q_eval.load_checkpoint(os.path.join(model_root_folder_path,'q_eval.chkpt'))
            self.q_next.load_checkpoint(os.path.join(model_root_folder_path,'q_next.chkpt'))
        except Exception as msg:
            print(msg)



    def store_experience(self, state : np.array, action : int, reward : float, n_state : np.array, done : bool):
        """Stores experience into the replay buffer

        Args:
            state (np.array): State
            action (int): action
            reward (float): reward
            n_state (np.array): next state
            done (bool): Terminal state?
        """
        self.replay_buffer.store_transition(state, action, reward, n_state, done)
        self.exp_counter+=1

    def get_action_idizes_greater_then_threshold(self, probs, prob_threshold):
        action_indexes = []

        for i, prob in enumerate(probs):
            if prob > prob_threshold:
                action_indexes.append(i)
        return action_indexes

    def select_action(self, state : np.ndarray, deploy=False) -> int:
        """Select an action based on the current state.

        Args:
            state (np.ndarray): Current state
            deploy (bool, optional): If true, always choose best action. Defaults to False.

        Returns:
            [type]: action
        """
        action_probs = self.q_eval.forward(state)
        m = Categorical(action_probs)
        action = m.sample()
        return int(action.item())


    def stochastic_act(self, state, prob_threshold):
        #state = np.array(state)
        #state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        probs = self.q_eval.forward(state)
        idizes = self.get_action_idizes_greater_then_threshold(probs.tolist(), prob_threshold), probs[0]
        # If all probs below prob_threshold
        if len(idizes) == 0:
            # Get action index with the largest probability
            action_idx = probs.argmax().cpu().item()
            idizes.append(action_idx)
        return idizes

    def model_checking_select_action(self, state : np.ndarray, prob_threshold):
        return self.stochastic_act(state, prob_threshold)

    def get_action_name_probability(self, env, action, state):
        # Action idx
        action_idx = env.action_mapper.action_name_to_action_index(action)
        # Action probability
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        probs = self.q_eval.forward(state).cpu()

        probs = probs.tolist()

        return probs[action_idx]


    def replace_target_network(self):
        """
        Replace the target network.
        """
        if self.learn_step_counter % self.replace == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def step_learn(self):
        """
        Agent learning.
        """
        if self.exp_counter<self.batch_size:
            return
        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()
        state_batch, action_batch, reward_batch, n_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        indices = torch.arange(0,self.batch_size).long()
        action_batch = action_batch.long()
        q_pred = self.q_eval.forward(state_batch)[indices, action_batch]
        q_next = self.q_next.forward(n_state_batch).max(dim=1).values.to(DEVICE)
        q_next[done_batch] = 0
        q_target = reward_batch.to(DEVICE) + self.gamma * q_next
        loss = self.q_eval.loss(q_target, q_pred)
        loss.backward()
        self.q_eval.optimizer.step()



