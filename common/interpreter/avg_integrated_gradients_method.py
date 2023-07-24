import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from torch.autograd import Variable
from common.interpreter.interpreter import *

class AvgIntegratedGradientImportance(Interpreter):

    def __init__(self, config):
        self.config = config

    # Integrated Gradients for feature importance analysis
    def __integrated_gradients(self, X, model, steps=50):
        # numpy float64 to float32
        X = X.astype(np.float32)

        # Calculate the baseline as the mean of the input array
        baseline = np.mean(X, axis=0, keepdims=True).astype(np.float32)
        baseline = torch.from_numpy(baseline).float()

        integrated_gradients_sum = None

        for i, x in enumerate(X):
            # numpy to torch
            x = torch.from_numpy(x).float()
            inputs = x.unsqueeze(0)

            # Create a linear interpolation between the baseline and inputs
            alphas = torch.linspace(0, 1, steps)[:, None, None]
            interpolated = baseline + alphas * (inputs - baseline)

            # Calculate gradients
            interpolated = Variable(interpolated, requires_grad=True)
            logits = model(interpolated)
            logits = logits.sum()
            logits.backward()

            # Calculate integrated gradients
            grads = interpolated.grad.data.mean(0)
            integrated_gradients_instance = (inputs - baseline) * grads

            if integrated_gradients_sum is None:
                integrated_gradients_sum = integrated_gradients_instance
            else:
                integrated_gradients_sum += integrated_gradients_instance

        # Average integrated gradients across all input instances
        integrated_gradients_avg = integrated_gradients_sum / len(X)
        # Calculate the relative feature rank based on the importance values
        feature_rank = torch.argsort(integrated_gradients_avg, descending=True)
        return integrated_gradients_avg, feature_rank


    def interpret(self, env, m_project, model_checking_info):
        X = np.array(model_checking_info['collected_states'])
        integrated_gradients, feature_rank = self.__integrated_gradients(X, m_project.agent.q_eval)
        print(integrated_gradients)
        feature_ranks = feature_rank.flatten().tolist()
        print(feature_ranks)
        for feature_idx, value in enumerate(feature_ranks):
            print(env.storm_bridge.state_mapper.inverse_mapping(feature_idx), ":", value)


