import mlflow
import os
import shutil
from typing import List
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.rl_agents.agent import Agent
from common.utilities.helper import *
from collections import OrderedDict
import torch
import numpy as np
import copy

def get_ranked_value(tensor, rank):
    # Check if the provided rank is valid
    if rank < 0 or rank >= tensor.size(0):
        raise ValueError("Rank must be within the range of the tensor size")

    # Get the sorted indices of the tensor in descending order
    sorted_indices = torch.argsort(tensor, descending=True)

    # Find the index corresponding to the specified rank
    rank_index = sorted_indices[rank]

    # Return the value at the specified rank index
    return tensor[rank_index].item(), rank_index.item()


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

    def copy(self):
        """Creates a deep copy of the replay buffer."""
        new_buffer = ReplayBuffer(self.size, self.state_memory.shape[1])
        new_buffer.memory_counter = self.memory_counter
        new_buffer.state_memory = np.copy(self.state_memory)
        new_buffer.new_state_memory = np.copy(self.new_state_memory)
        new_buffer.action_memory = np.copy(self.action_memory)
        new_buffer.reward_memory = np.copy(self.reward_memory)
        new_buffer.terminal_memory = np.copy(self.terminal_memory)
        return new_buffer

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
                else:
                    x = F.relu(self.layers[i](x))
            return x
        except:
            state = torch.tensor(state).float().to(DEVICE)
            x = state
            for i in range(len(self.layers)):
                if i == (len(self.layers)-1):
                    x = self.layers[i](x)
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

    def copy(self):
        """Creates a deep copy of the network."""
        # Create a new network instance with the same initialization parameters
        new_network = DeepQNetwork(
            state_dimension=self.state_dimension,
            number_of_neurons=self.number_of_neurons,
            number_of_actions=self.number_of_actions,
            lr=self.learning_rate
        )
        # Copy over the state dict (weights and biases)
        new_network.load_state_dict(copy.deepcopy(self.state_dict()))
        # Copy the optimizer state
        new_network.optimizer.load_state_dict(copy.deepcopy(self.optimizer.state_dict()))
        return new_network



class DQNAgent(Agent):

    def __init__(self, state_dimension : int, number_of_neurons, number_of_actions : List[int], epsilon : float =1, epsilon_dec : float =0.99999, epsilon_min : float = 0.1, gamma : float = 0.99, learning_rate : float = 0.001, replace : int =100, batch_size : int =64, replay_buffer_size : int =10000):
        """Initialize Deep Q-Learning Agent

        Args:
            state_dimension (np.array): State Dimension
            number_of_neurons (List[int]): Each element is a new layer and defines the number of neurons in the current layer
            number_of_actions (int): Number of Actions
            epsilon (float, optional): Init epislon value. Defaults to 1.
            epsilon_dec (float, optional): Epsilon decay value. Defaults to 0.99999.
            epsilon_min (float, optional): Epsilon minimum value. Defaults to 0.1.
            gamma (float, optional): Gamma value. Defaults to 0.99.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            replace (int, optional): Replace Interval of the target network (q_next). Defaults to 100.
            batch_size (int, optional): Batch Size. Defaults to 64.
            replay_buffer_size (int, optional): Replay Buffer size. Defaults to 10000.
        """
        self.number_of_actions = number_of_actions
        self.replay_buffer = ReplayBuffer(replay_buffer_size, state_dimension)
        self.state_dimension = state_dimension
        self.number_of_neurons = number_of_neurons
        self.learning_rate = learning_rate
        self.q_eval = DeepQNetwork(state_dimension, number_of_neurons, number_of_actions, learning_rate)
        self.q_next = DeepQNetwork(state_dimension, number_of_neurons, number_of_actions, learning_rate)
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.gamma = torch.tensor(gamma).to(DEVICE)
        self.replace = replace
        self.batch_size = batch_size
        self.exp_counter = 0
        self.learn_step_counter = 0
        self.nth_action_instead = None
        self.block_nth_action = None
        self.state_filters = []
        self.replay_buffer_size = replay_buffer_size

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
            self.q_eval_path = os.path.join(model_root_folder_path,'q_eval.chkpt')
            self.q_eval.load_checkpoint(self.q_eval_path)
            self.q_next_path = os.path.join(model_root_folder_path,'q_next.chkpt')
            self.q_next.load_checkpoint(self.q_next_path)
        except Exception as msg:
            print(msg)
    
    def load2(self):
        """Loads the Agent from the MLFlow server.

        Args:
            model_root_folder_path (str): Model root folder path.
        """
        try:
            self.q_eval.load_checkpoint(self.q_eval_path)
            self.q_next.load_checkpoint(self.q_next_path)
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


    def get_nt_action_alternative(self, state : np.array, nth_action_instead):
        """Get action index from the action name

        Args:
            state (np.array): State
            action_name (str): Action Name

        Returns:
            int: Action Index
        """
        _, action_index = get_ranked_value(self.q_eval.forward(state), nth_action_instead)
        return action_index

    def select_action(self, state : np.ndarray, deploy=False) -> int:
        """Select random action or action based on the current state.

        Args:
            state (np.ndarray): Current state
            deploy (bool, optional): If true, no random states. Defaults to False.

        Returns:
            [type]: action
        """
        if self.block_nth_action is not None:
            if self.block_nth_action == torch.argmax(self.q_eval.forward(state)).item():
                if self.nth_action_instead is not None:
                    _, action_index = get_ranked_value(self.q_eval.forward(state), self.nth_action_instead)
                    return action_index
                else:
                    # Return the second best action
                    _, action_index = get_ranked_value(self.q_eval.forward(state), 1)
                    return action_index
        if self.nth_action_instead is not None:
            _, action_index = get_ranked_value(self.q_eval.forward(state), self.nth_action_instead)
            return action_index
        if deploy:
            for state_filter in self.state_filters:
                #print("====================================")
                #print("State Filter", state_filter)
                #print("State Copy", state_copy)
                #print("Original State", state)
                #print(state_filter[0].__class__.__name__, state.__class__.__name__)
                if np.array_equal(state_filter[0], state):
                    return state_filter[1] # For alternative checking
            return int(torch.argmax(self.q_eval.forward(state)).item())
        if torch.rand(1).item() < self.epsilon:
            self.epsilon *= self.epsilon_dec
            self.epsilon = max(self.epsilon, self.epsilon_min)
            return int(torch.randint(0,self.number_of_actions,(1,)).item())
        else:
            return int(torch.argmax(self.q_eval.forward(state)).item())

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

    def copy(self):
        """Creates a deep copy of the agent."""
        # Create a new agent with the same initialization parameters
        new_agent = DQNAgent(
            state_dimension=self.state_dimension,
            number_of_neurons=self.number_of_neurons,
            number_of_actions=self.number_of_actions,
            epsilon=self.epsilon,
            epsilon_dec=self.epsilon_dec,
            epsilon_min=self.epsilon_min,
            gamma=self.gamma.item(),  # Convert tensor to float
            learning_rate=self.learning_rate,
            replace=self.replace,
            batch_size=self.batch_size,
            replay_buffer_size=self.replay_buffer_size
        )

        # Copy over the networks
        new_agent.q_eval.load_state_dict(copy.deepcopy(self.q_eval.state_dict()))
        new_agent.q_next.load_state_dict(copy.deepcopy(self.q_next.state_dict()))

        # Copy the optimizer states
        new_agent.q_eval.optimizer.load_state_dict(copy.deepcopy(self.q_eval.optimizer.state_dict()))
        new_agent.q_next.optimizer.load_state_dict(copy.deepcopy(self.q_next.optimizer.state_dict()))

        # Copy over other attributes
        new_agent.exp_counter = self.exp_counter
        new_agent.learn_step_counter = self.learn_step_counter
        new_agent.nth_action_instead = self.nth_action_instead
        new_agent.block_nth_action = self.block_nth_action
        new_agent.state_filters = copy.deepcopy(self.state_filters)
        new_agent.epsilon = self.epsilon
        new_agent.epsilon_dec = self.epsilon_dec
        new_agent.epsilon_min = self.epsilon_min
        new_agent.gamma = torch.tensor(self.gamma.item()).to(DEVICE)

        # Copy
        new_agent.q_eval_path = self.q_eval_path
        new_agent.q_next_path = self.q_next_path

        # Deep copy the replay buffer
        new_agent.replay_buffer = self.replay_buffer.copy()

        return new_agent
