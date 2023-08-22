# This PPO code is a modified version of the PPO code from https://github.com/nikhilbarhate99/PPO-PyTorch
# Credits to nikhilbarhate99
import torch
import torch.nn as nn
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
################################## set device ##################################

# set device to cpu or cuda
DEVICE = torch.device('cpu')



################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()


        self.actor = nn.Sequential(
                            nn.Linear(state_dim, 1028),
                            nn.Tanh(),
                            nn.Linear(1028, 1028),
                            nn.Tanh(),
                            nn.Linear(1028, action_dim),
                            nn.Softmax(dim=-1)
                        )
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 1028),
                        nn.Tanh(),
                        nn.Linear(1028, 1028),
                        nn.Tanh(),
                        nn.Linear(1028, 1)
                    )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)


        dist = Categorical(action_probs)


        action = dist.sample()

        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)

        return action.detach(), action_logprob.detach(), state_val.detach(), action_probs

    def evaluate(self, state, action):
        # Reshape to batch
        if len(state.shape) == 1:
            state = torch.squeeze(state).unsqueeze(-1)


        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy




class PPO(StochasticAgent):
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, old):
        self.old = old
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def store_experience(self, state : np.array, action : int, reward : float, n_state : np.array, done : bool):
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)



    def select_action(self, state : np.ndarray, deploy : bool =False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(DEVICE)
            action, action_logprob, state_val, action_probs = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        if self.old and deploy:
            # Get action index with the highest probability
            action_idx = torch.argmax(action_probs).item()
            return action_idx

        return action.item()

    def stochastic_act(self, state, prob_threshold):
        with torch.no_grad():
            state = np.array(state)
            state = torch.FloatTensor(state).to(DEVICE)
            action, action_logprob, state_val, action_probs = self.policy_old.act(state)
        return self.get_action_idizes_not_zero(action_probs.tolist(), prob_threshold), action_probs.tolist()

    def get_action_idizes_not_zero(self, probs, prob_threshold):
        action_indexes = []
        for i, prob in enumerate(probs):
            if prob > prob_threshold:
                action_indexes.append(i)
        return action_indexes

    def model_checking_select_action(self, state : np.ndarray, prob_threshold):
        return self.stochastic_act(state, prob_threshold)

    def get_action_name_probability(self, env, action, state):
        # Action idx
        action_idx = env.action_mapper.action_name_to_action_index(action)
        # Action probability
        state = torch.from_numpy(np.array([state])).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            state = torch.FloatTensor(state).to(DEVICE)
            action, action_logprob, state_val, action_probs = self.policy_old.act(state)
        # probs to float
        try:
            return action_probs[0][action_idx].item()
        except:
            return action_probs[0][0][action_idx].item()


    def episodic_learn(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(DEVICE)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(DEVICE)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(DEVICE)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(DEVICE)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()


    def save(self, artifact_path='model'):
        """
        Saves the agent onto the MLFLow Server.
        """
        try:
            os.mkdir('tmp_model')
        except Exception as msg:
            pass
        torch.save(self.policy_old.state_dict(), 'tmp_model/policy.chkpt')
        mlflow.log_artifacts("tmp_model", artifact_path=artifact_path)
        shutil.rmtree('tmp_model')

    def load(self, model_root_folder_path :str):
        """Loads the Agent from the MLFlow server.

        Args:
            model_root_folder_path (str): Model root folder path.
        """
        try:
            checkpoint_path = os.path.join(model_root_folder_path,'policy.chkpt')
            self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
            self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        except:
            pass




