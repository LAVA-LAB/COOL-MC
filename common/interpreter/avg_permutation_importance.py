from sklearn.metrics import accuracy_score
import numpy as np
import torch
from common.interpreter.interpreter import *
# Partitioned Permutation Importance for feature importance analysis
# For each feature assignment range of feature f, we create a feature importance vector
class AvgPermutationImportance(Interpreter):

    def __init__(self, config):
        self.config = config

    # Function to calculate permutation importance
    def permutation_importance(self, model, X, y, metric, num_permutations=30):
        baseline_score = metric(y, model(X).argmax(dim=1).detach().numpy())
        feature_importances = np.zeros(X.shape[1])

        for i in range(X.shape[1]):
            permuted_scores = []
            for _ in range(num_permutations):
                X_permuted = X.clone().detach()
                X_permuted[:, i] = X_permuted[:, i][torch.randperm(X.shape[0])]
                permuted_score = metric(y, model(X_permuted).argmax(dim=1).detach().numpy())
                permuted_scores.append(permuted_score)

            feature_importances[i] = baseline_score - np.mean(permuted_scores)

        return feature_importances

    def interpret(self, env, m_project, model_checking_info):
        X = np.array(model_checking_info['collected_states'])
        # to torch
        X = torch.from_numpy(X).float()
        y = np.array(model_checking_info['collected_action_idizes'])
        # to torch
        y = torch.from_numpy(y).long()

        permutation_importances = self.permutation_importance(m_project.agent.q_eval, X, y, accuracy_score)
        for feature_idx, value in enumerate(permutation_importances):
            print(env.storm_bridge.state_mapper.inverse_mapping(feature_idx), ":", value)
