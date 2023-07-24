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
import matplotlib.colors as colors
#import plotly.graph_objects as go

import pygraphviz as pgv
from networkx.drawing.nx_agraph import to_agraph


def heatmap(model, original_weights, layer_index):
    FONT_SIZE = 18
    if layer_index < 0 or layer_index >= len(model.layers):
        raise ValueError(f"Invalid layer index. Expected between 0 and {len(model.layers)-1}, got {layer_index}")

    # Add a small random noise to weights to break ties
    #model.layers[layer_index].weight.data.add_(torch.randn_like(model.layers[layer_index].weight.data) * 1e-5)

    pruned_weights = model.layers[layer_index].weight.detach().clone()

    heatmap = np.zeros_like(original_weights)
    heatmap[(original_weights != 0) & (pruned_weights == 0)] = 1  # pruned
    heatmap[(original_weights != 0) & (pruned_weights != 0)] = 2  # unpruned

    fig = plt.figure(figsize=(10, 1))

    # Define custom color map
    cmap = colors.ListedColormap(['blue', 'red', 'black'])

    plt.xticks(fontsize=FONT_SIZE)  # Adjust to desired font size
    plt.yticks(fontsize=FONT_SIZE)  # Adjust to desired font size
    plt.imshow(heatmap.T, cmap=cmap, vmin=0, vmax=2)
    #plt.title(f'Layer {layer_index} Pruning Heatmap')
    plt.ylabel("Input")
    plt.xlabel("Output")

    # Disable tick labels
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()


    plt.savefig(f'pruning_heatmap_layer_{layer_index}.png', format='png', dpi=300)
    plt.savefig(f'pruning_heatmap_layer_{layer_index}.eps', dpi=300, bbox_inches='tight', format='eps')
    plt.close()  # to free memory

    # Calculate percentages of pruned and not pruned weights
    total_weights = np.prod(heatmap.shape)
    pruned_weights = np.sum(heatmap == 1)
    unpruned_weights = np.sum(heatmap == 2)
    pruned_weights_percent = pruned_weights / total_weights * 100
    unpruned_weights_percent = unpruned_weights / total_weights * 100

    print(f"Percentage of pruned weights: {pruned_weights_percent}%")
    print(f"Percentage of unpruned weights: {unpruned_weights_percent}%")


def get_original_weights(model,layer_index):
    # Save a copy of original weights before pruning for all layers
    original_weights = model.layers[layer_index].weight.detach().clone()
    return original_weights
