import torch.nn.utils.prune as prune
from typing import List
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors


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
            state = torch.tensor(state).float()
            x = state
            for i in range(len(self.layers)):
                if i == (len(self.layers)-1):
                    x = self.layers[i](x)
                else:
                    x = F.relu(self.layers[i](x))
            return x



def prune_and_heatmap(model, layer_index: int, neuron_indices: List[int]):
    FONT_SIZE = 18
    if layer_index < 0 or layer_index >= len(model.layers):
        raise ValueError(f"Invalid layer index. Expected between 0 and {len(model.layers)-1}, got {layer_index}")

    original_weights = model.layers[layer_index].weight.detach().clone()

    # Create a binary mask of the same size as the layer weights
    mask = torch.ones_like(original_weights)
    # Set all outgoing connections from the neuron at position i to be pruned
    for neuron_index in neuron_indices:
        mask[:, neuron_index] = 0

    prune.custom_from_mask(model.layers[layer_index], name="weight", mask=mask)
    prune.remove(model.layers[layer_index], name="weight")

    pruned_weights = model.layers[layer_index].weight.detach().clone()

    heatmap = np.zeros_like(original_weights)
    heatmap[(original_weights != 0) & (pruned_weights == 0)] = 1  # pruned
    heatmap[(original_weights != 0) & (pruned_weights != 0)] = 2  # unpruned


    plt.figure(figsize=(10, 10))

    # Define custom color map
    cmap = colors.ListedColormap(['blue', 'red', 'black'])

    plt.xticks(fontsize=FONT_SIZE)  # Adjust to desired font size
    plt.yticks(fontsize=FONT_SIZE)  # Adjust to desired font size
    plt.imshow(heatmap, cmap=cmap, vmin=0, vmax=2)
    plt.title(f'Layer {layer_index + 1} Pruning Heatmap')
    plt.xlabel("Input Neurons")  # x-axis label
    plt.ylabel("Output Neurons")  # y-axis label
    plt.savefig(f'pruning_heatmap_layer_{layer_index+1}.png', format='png', dpi=300)
    plt.savefig(f'pruning_heatmap_layer_{layer_index+1}.eps', dpi=300, bbox_inches='tight', format='eps')
    plt.close()  # to free memory

    # Calculate percentages of pruned and not pruned weights
    total_weights = np.prod(heatmap.shape)
    pruned_weights = np.sum(heatmap == 1)
    unpruned_weights = np.sum(heatmap == 2)
    pruned_weights_percent = pruned_weights / total_weights * 100
    unpruned_weights_percent = unpruned_weights / total_weights * 100

    print(f"Percentage of pruned weights: {pruned_weights_percent}%")
    print(f"Percentage of unpruned weights: {unpruned_weights_percent}%")



# Initialize the model
model = DeepQNetwork(4, [3,10,30], 3, 0.0001)

# Prune and plot
prune_and_heatmap(model, 0, [0, 2])

