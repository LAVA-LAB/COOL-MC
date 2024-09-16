from common.interpreter.interpreter import Interpreter

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

class CriticalStateEvaluator(Interpreter):

    def __init__(self, config):
        self.config = config

    def interpret(self, env, m_project, model_checking_info):
        m_project.preprocessors[0].collect = False
        states = []
        rank_idx = []
        rank_val = []
        original_mdp_result = model_checking_info['mdp_reward_result']
        original_results = [] # state-mdp_reward pairs
        other_results = []
        # For critical state pair
        for idx, critical_pair in enumerate(m_project.preprocessors[0].all_critical_states):
            state = critical_pair[0]
            states.append(state)
            state_importance = critical_pair[1]
            print("State:", state, "State Importance:", state_importance)
            m_project.preprocessors[0].critical_state = state
            mdp_reward_result, model_checking_info = env.storm_bridge.model_checker.induced_markov_chain(m_project.agent, m_project.preprocessors, env, m_project.command_line_arguments['constant_definitions'], m_project.command_line_arguments['prop'], False)
            print("State:", state, "State Importance:", state_importance, "MDP Reward Result:", mdp_reward_result)
            rank_idx.append(idx)
            rank_val.append(mdp_reward_result)
            if mdp_reward_result == original_mdp_result:
                original_results.append((state, "original"))
            else:
                other_results.append((state, "other"))
            
        
        # Plot rank index and value
        plt.figure(figsize=(10, 5))
        plt.plot(rank_idx, rank_val)
        plt.xlabel('Rank Index')
        plt.ylabel('Rank Value')
        plt.title('Rank Index vs Rank Value')
        plt.savefig('rank_plot.png')

        # Perform PCA for critical states plot
        states = np.array(states)
        if states.ndim == 1:
            states = states.reshape(-1, 1)  # Ensure states are in the correct shape

        # Standardize the data
        states_standardized = StandardScaler().fit_transform(states)

        # Apply PCA
        pca = PCA(n_components=2)  # Reduce to 2 components for visualization
        principal_components = pca.fit_transform(states_standardized)
        
        plt.figure(figsize=(10, 5))
        plt.scatter(principal_components[:, 0], principal_components[:, 1], c=rank_val, cmap='viridis')
        plt.colorbar(label='MDP Reward Result')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA of Critical States')
        plt.savefig('pca_plot.png')

        

# Assuming the necessary environment and project setup
# config = ...
# env = ...
# m_project = ...
# model_checking_info = ...

# evaluator = CriticalStateEvaluator(config)
# evaluator.interpret(env, m_project, model_checking_info)
