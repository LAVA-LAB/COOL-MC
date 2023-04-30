import numpy as np
import pickle
import os
from common.rl_agents.agent import DeterministicAgent
import os
import mlflow
import shutil

class SarsaMaxAgent(DeterministicAgent):

    def __init__(self, number_of_actions=6, epsilon=0.5, epsilon_dec=0.9999, epsilon_min=0.01, alpha=0.6, gamma = 0.9):
        super().__init__()
        self.number_of_actions = number_of_actions
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.alpha = alpha
        self.Q = dict()



    def select_action(self, state, deploy=False):
        state, next_state =  self.__make_sure_that_states_in_dictionary(state)
        action_index = None
        if deploy == True:
            action_index = np.argmax(self.Q[str(state)])
            return int(action_index)
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_dec
        if np.random.rand() < self.epsilon:
            # Random Choice
            action_index = np.random.choice(self.number_of_actions)
        else:
            # Greedy Choice
            action_index = np.argmax(self.Q[str(state)])
        return int(action_index)


    def __make_sure_that_states_in_dictionary(self, state, next_state=None):
        state = str(state)
        if state not in self.Q.keys():
            self.Q[state] = [0]*self.number_of_actions
        if next_state is not None:
            next_state = str(next_state)
            if next_state not in self.Q.keys():
                self.Q[next_state] = [0]*self.number_of_actions
        return state, next_state

    def store_experience(self, state, action, reward, next_state, terminal):
        '''
        Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - terminal: whether the episode is complete (True or False)
        '''
        state, next_state = self.__make_sure_that_states_in_dictionary(state, next_state)
        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * max(self.Q[next_state]) - self.Q[state][action])

    def get_hyperparameters(self):
        '''
        Get hyperparameters
        :return hyperparameters
        '''
        h_params = dict()
        h_params['number_of_actions'] = self.number_of_actions
        h_params['epsilon'] = self.epsilon
        h_params['epsilon_dec'] = self.epsilon_dec
        h_params['epsilon_min'] = self.epsilon_min
        h_params['gamma'] = self.gamma
        h_params['alpha'] = self.alpha
        h_params['Q'] =self.Q
        return h_params

    def load(self, root_folder):
        try:
            with open(os.path.join(root_folder, 'model.pickle'), 'rb') as agent_file:
                config_dict = pickle.load(agent_file)
                self.Q = config_dict["Q"]
        except:
            pass




    def save(self):
        try:
            os.mkdir('tmp_model')
        except:
            pass
        config_dict = {
            'Q': self.Q
        }
        with open(os.path.join('tmp_model', 'model.pickle'), 'wb') as agent_file:
            pickle.dump(config_dict, agent_file)
        mlflow.log_artifacts("tmp_model", artifact_path="model")

        shutil.rmtree('tmp_model')


    def load(self, model_root_folder_path):
        with open(os.path.join(model_root_folder_path,'model.pickle'), 'rb') as agent_file:
                config_dict = pickle.load(agent_file)
                self.Q = config_dict["Q"]
        try:
            with open(os.path.join(model_root_folder_path,'model.pickle'), 'rb') as agent_file:
                config_dict = pickle.load(agent_file)
                self.Q = config_dict["Q"]
        except:
            pass
