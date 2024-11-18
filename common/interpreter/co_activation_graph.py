import torch
import torch.nn.functional as F
import stormpy
import torch
import numpy as np
import pandas as pd
import community  # This is the python-louvain package
import re
import matplotlib.pyplot as plt
import itertools
import networkx as nx




def create_coactivation_graph(agent, states):


    # Ensure states is a numpy array
    observations = np.array(states)
    # Convert observations to a tensor
    obs_tensor = torch.tensor(observations).float().to(agent.q_eval.DEVICE)

    # Prepare to record activations
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            if name not in activations:
                activations[name] = []
            # For compatibility, we need to detach and move to CPU
            activations[name].append(output.detach().cpu().numpy())
        return hook

    # Register hooks to capture activations from the network layers
    current_idx = 1  # Start from 1 since layer_0 is the input
    hook_handles = []  # List to keep track of hook handles
    for idx, layer in enumerate(agent.q_eval.layers):
        if isinstance(layer, torch.nn.Linear):
            handle = layer.register_forward_hook(get_activation(f"layer_{current_idx}"))
            hook_handles.append(handle)
            current_idx += 1

    # Include input observations in activations
    activations["layer_0"] = observations

    # Pass observations through the network to collect activations
    with torch.no_grad():
        _ = agent.q_eval(obs_tensor)

    # **Remove hooks after use**
    for handle in hook_handles:
        handle.remove()

    # Process activations and compute co-activation graphs
    # First, ensure all activations are in the form (num_samples, num_neurons)
    activation_arrays = {}
    for layer_name, layer_activations in activations.items():
        if layer_name == "layer_0":
            # Observations already in correct shape
            activation_arrays[layer_name] = layer_activations
        else:
            # Concatenate activations into a numpy array
            layer_activations = np.concatenate(layer_activations, axis=0)
            activation_arrays[layer_name] = layer_activations

    # Check for zero variance columns and remove them
    for layer_name in list(activation_arrays.keys()):
        variances = np.var(activation_arrays[layer_name], axis=0)
        zero_variance_indices = np.where(variances == 0)[0]
        if len(zero_variance_indices) > 0:
            print(f"Removing zero variance neurons from {layer_name}: indices {zero_variance_indices}")
            activation_arrays[layer_name] = np.delete(activation_arrays[layer_name], zero_variance_indices, axis=1)

    # Combine all activations into a single array
    all_activations = np.concatenate(
        [activation_arrays[layer_name] for layer_name in activation_arrays],
        axis=1
    )

    # Create column names
    column_names = []
    for layer_name in activation_arrays:
        num_neurons = activation_arrays[layer_name].shape[1]
        for neuron_idx in range(num_neurons):
            column_names.append(f"{layer_name}_neuron_{neuron_idx}")

    # Create a DataFrame with all activations
    activation_df = pd.DataFrame(all_activations, columns=column_names)

    # Compute the correlation matrix (co-activation graph)
    co_activation_matrix = activation_df.corr()

    # Replace NaN values with zeros (if any remain)
    co_activation_matrix = co_activation_matrix.fillna(0)

    # Print the co-activation DataFrame
    print("\nCo-activation matrix including inputs and outputs:")
    print(co_activation_matrix)

    return co_activation_matrix



def analyze_community_with_modularity(co_activation_matrix):
    """
    Performs community analysis on the co-activation graph using NetworkX's Louvain method and computes modularity.

    Parameters:
    - co_activation_matrix: pandas DataFrame representing the co-activation matrix.

    Returns:
    - A tuple containing:
        - A list of tuples (neuron_name, community_label) sorted by community.
        - The modularity score of the partition.
    """

    # Create an undirected graph from the co-activation matrix
    G = nx.Graph()

    # Get the list of neurons
    neurons = co_activation_matrix.columns.tolist()

    # Add nodes to the graph
    G.add_nodes_from(neurons)

    # Add weighted edges to the graph based on co-activation (correlation)
    for i in range(len(neurons)):
        for j in range(i + 1, len(neurons)):
            weight = co_activation_matrix.iloc[i, j]
            if not np.isnan(weight) and weight > 0:
                G.add_edge(neurons[i], neurons[j], weight=weight)

    # Perform community detection using NetworkX's Louvain method
    # Note: louvain_communities is available in NetworkX 2.5 and above
    communities = nx.algorithms.community.louvain_communities(G, weight='weight', seed=42)

    # Assign community labels
    community_labels = {}
    for idx, community in enumerate(communities):
        for neuron in community:
            community_labels[neuron] = idx

    # Compute modularity
    modularity = nx.algorithms.community.modularity(G, communities, weight='weight')

    # Convert the community labels to a sorted list of tuples
    neuron_communities = sorted(
        [(neuron, community_labels[neuron]) for neuron in neurons if "Unnamed: 0" not in neuron],
        key=lambda x: (x[1], x[0])
    )

    return neuron_communities, modularity



def analyze_community2(co_activation_matrix):
    """
    Performs community analysis on the co-activation graph using NetworkX's Louvain method.

    Parameters:
    - co_activation_matrix: pandas DataFrame representing the co-activation matrix.

    Returns:
    - A list of tuples (neuron_name, community_label) sorted by community.
    """
    

    # Create an undirected graph from the co-activation matrix
    G = nx.Graph()

    # Get the list of neurons
    neurons = co_activation_matrix.columns.tolist()

    # Add nodes to the graph
    G.add_nodes_from(neurons)

    # Add weighted edges to the graph based on co-activation (correlation)
    for i in range(len(neurons)):
        for j in range(i + 1, len(neurons)):
            weight = co_activation_matrix.iloc[i, j]
            if not np.isnan(weight) and weight > 0:
                G.add_edge(neurons[i], neurons[j], weight=weight)

    # Perform community detection using NetworkX's Louvain method
    # Note: louvain_communities is available in NetworkX 2.5 and above
    communities = nx.algorithms.community.louvain_communities(G, weight='weight', seed=42)

    # Assign community labels
    community_labels = {}
    for idx, community in enumerate(communities):
        for neuron in community:
            community_labels[neuron] = idx

    # Convert the community labels to a sorted list of tuples
    neuron_communities = sorted(
        [(neuron, community_labels[neuron]) for neuron in neurons if "Unnamed: 0" not in neuron],
        key=lambda x: (x[1], x[0])
    )

    return neuron_communities



def analyze_community(co_activation_matrix):
    """
    Performs community analysis on the co-activation graph using NetworkX's Greedy Modularity method.

    Parameters:
    - co_activation_matrix: pandas DataFrame representing the co-activation matrix.

    Returns:
    - A list of tuples (neuron_name, community_label) sorted by community.
    """
    # Create an undirected graph from the co-activation matrix
    G = nx.Graph()

    # Get the list of neurons
    neurons = co_activation_matrix.columns.tolist()

    # Add nodes to the graph
    G.add_nodes_from(neurons)

    # Add weighted edges to the graph based on co-activation (correlation)
    for i in range(len(neurons)):
        for j in range(i + 1, len(neurons)):
            weight = co_activation_matrix.iloc[i, j]
            if not np.isnan(weight) and weight > 0:
                G.add_edge(neurons[i], neurons[j], weight=weight)

    # Perform community detection using the Greedy Modularity method
    communities = nx.algorithms.community.greedy_modularity_communities(G, weight='weight')
    

    # Assign a unique community label to each community
    community_labels = {}
    for idx, community in enumerate(communities):
        for neuron in community:
            community_labels[neuron] = idx

    # Convert the community labels to a sorted list of tuples
    neuron_communities = sorted(
        [(neuron, community_labels[neuron]) for neuron in neurons],
        key=lambda x: (x[1], x[0])
    )
    # Remove tuple that has name "Unnamed"
    neuron_communities = [neuron for neuron in neuron_communities if "Unnamed: 0" not in neuron[0]]
    return neuron_communities


def analyze_centrality2(co_activation_matrix):
    """
    Performs centrality analysis on the co-activation graph using PageRank centrality.

    Parameters:
    - co_activation_matrix: pandas DataFrame representing the co-activation matrix.

    Returns:
    - A tuple containing:
      - sorted_neurons: A list of tuples (neuron_name, centrality_score) sorted from most to least important neuron.
      - input_feature_ranking: A list of tuples for neurons in the input layer.
      - action_output_ranking: A list of tuples for neurons in the output layer.
    """


    # Extract the number of layers
    last_neuron_name = co_activation_matrix.columns[-1]
    number_of_layers = int(last_neuron_name.split('_')[1])

    # Create an undirected graph from the co-activation matrix
    G = nx.Graph()

    # Get the list of neurons
    neurons = co_activation_matrix.columns.tolist()

    # Add nodes to the graph
    G.add_nodes_from(neurons)

    # Add weighted edges to the graph
    for i in range(len(neurons)):
        for j in range(i + 1, len(neurons)):
            weight = co_activation_matrix.iloc[i, j]
            if not np.isnan(weight) and weight != 0:
                # Use absolute value to ensure positive weights for centrality measures
                G.add_edge(neurons[i], neurons[j], weight=abs(weight))

    # Compute centrality scores using PageRank
    centrality_scores = nx.pagerank(G, weight='weight')

    # Sort neurons based on centrality scores
    sorted_neurons = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)

    # Extract rankings for input features and output actions
    input_feature_ranking = [neuron for neuron in sorted_neurons if 'layer_0_neuron' in neuron[0]]
    action_output_ranking = [neuron for neuron in sorted_neurons if f'layer_{number_of_layers}_neuron' in neuron[0]]

    return sorted_neurons, input_feature_ranking, action_output_ranking



def analyze_centrality(co_activation_matrix):
    """
    Performs centrality analysis on the co-activation graph.

    Parameters:
    - co_activation_matrix: pandas DataFrame representing the co-activation matrix.

    Returns:
    - A list of tuples (neuron_name, centrality_score) sorted from most to least important neuron.
    """
    # Get last row index
    last_row_index = co_activation_matrix.shape[0] - 1
    # Extract the layer index
    number_of_layers = int(co_activation_matrix.columns[last_row_index].split('_')[1])
    # Create a graph from the co-activation matrix
    G = nx.Graph()
    
    # Get the list of neurons
    neurons = co_activation_matrix.columns.tolist()
    
    # Add nodes to the graph
    G.add_nodes_from(neurons)
    
    # Add weighted edges to the graph
    for i in range(len(neurons)):
        for j in range(i+1, len(neurons)):
            weight = co_activation_matrix.iloc[i, j]
            if not np.isnan(weight) and weight != 0:
                # Use absolute value to ensure positive weights for centrality measures
                G.add_edge(neurons[i], neurons[j], weight=abs(weight))
    
    # Compute centrality scores (e.g., eigenvector centrality)
    centrality_scores = nx.eigenvector_centrality_numpy(G, weight='weight')
    
    # Sort neurons based on centrality scores
    sorted_neurons = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)

    # Only keep input_feature in ranking
    input_feature_ranking = [neuron for neuron in sorted_neurons if 'layer_0_neuron' in neuron[0]]
    
    # Only keep action_ouput in ranking
    action_output_ranking = [neuron for neuron in sorted_neurons if f'layer_{number_of_layers}_neuron' in neuron[0]]
    return sorted_neurons, input_feature_ranking, action_output_ranking

def plot_layer_neuron_importance(*lists, filename='layer_neuron_importance.png', dataset_names=["Dataset 1", "Dataset 2"]):
    # Function to extract x, y from the string and importance
    def extract_values(data_list):
        x_values = []
        y_values = []
        importance_values = []
        for element in data_list:
            # Extract the x and y using regex
            match = re.match(r"layer_(\d+)_neuron_(\d+)", element[0])
            if match:
                x_values.append(int(match.group(1)))
                y_values.append(int(match.group(2)))
                importance_values.append(element[1])
        return x_values, y_values, importance_values

    # Predefined markers, can be extended
    markers = itertools.cycle(['o', '<', '>', '|', '_', 'x', '*', 's', '^', 'P', 'D', 'v', 'o', '<', '>', 'h'])

    # Initialize lists for all x, y, and importance values
    all_x = []
    all_y = []
    all_importance = []

    # Collect x, y, and importance values from each list
    for data_list in lists:
        x, y, importance = extract_values(data_list)
        all_x.append(x)
        all_y.append(y)
        all_importance.append(importance)

    # Flatten importance values and normalize them for color mapping
    all_importance_flat = [val for sublist in all_importance for val in sublist]
    norm_importance = (all_importance_flat - np.min(all_importance_flat)) / (np.max(all_importance_flat) - np.min(all_importance_flat))

    # Create the scatter plot
    plt.figure(figsize=(8, 6))

    # Start plotting each list with a different marker
    start = 0
    for i, (x, y, importance) in enumerate(zip(all_x, all_y, all_importance)):
        end = start + len(importance)
        norm_importance_subset = norm_importance[start:end]
        marker = next(markers)  # Get the next marker in the cycle
        # For x elements equal 0, multiply the corresponding y elements by 10
        #y = [val * 10 if x_val == 0 else val for x_val, val in zip(x, y)]
        # The same for the last layer
        #y = [val * 10 if x_val == max(x) else val for x_val, val in zip(x, y)]
        plt.scatter(x, y, c=norm_importance_subset, cmap='viridis', marker=marker, label=f'{dataset_names[i]}', s=100, alpha=0.6)
        start = end

    # Add a color bar
    plt.colorbar(label='Importance')
    # Only integer x and y ticks
    plt.xticks(np.arange(min(min(all_x)), max(max(all_x)) + 2, 1))
    plt.yticks(np.arange(min(min(all_y)), max(max(all_y)) + 1, 20))
    # Add labels and legend
    plt.xlabel('Layer Index')
    plt.ylabel('Neuron Index')
    plt.legend(loc='upper right')

    # Show plot
    plt.title('Layer vs Neuron with Importance Values')
    plt.grid(True)
    plt.savefig(filename)
    # Save to .eps format
    plt.savefig(filename.replace('.png', '.eps'), format='eps')
    plt.close()


def plot_neurons(data, filename):
    """
    Plots the neurons at their corresponding x-y coordinates with colors indicating their communities.
    Highlights the neurons in the last layer by multiplying their y-values by 10 and saves the plot to a file.
    
    Parameters:
    data (list of tuples): Each tuple should contain the neuron name and its community.
    filename (str): The path where the plot will be saved.
    """
    # Parse the data
    x_vals = []
    y_vals = []
    communities = []
    neuron_names = []

    for neuron, community in data:
        parts = neuron.split('_')
        layer = int(parts[1])  # x-coordinate: layer number
        neuron_index = int(parts[3])  # y-coordinate: neuron index in the layer
        x_vals.append(layer)
        y_vals.append(neuron_index)
        communities.append(community)
        neuron_names.append(neuron)

    # Assign colors based on community
    unique_communities = sorted(set(communities))  # Ensure the communities are sorted
    community_to_color = {community: i for i, community in enumerate(unique_communities)}
    norm = plt.Normalize(vmin=0, vmax=len(unique_communities) - 1)
    cmap = plt.cm.get_cmap('rainbow', len(unique_communities))
    
    # Map colors to the communities
    colors = [cmap(community_to_color[community]) for community in communities]

    # Find the maximum x value (last layer)
    max_x = max(x_vals)
    # Find the maximum y value
    max_y = max(y_vals)


    # Modify y_vals for neurons in the first and last layer
    adjusted_y_vals = []
    for x, y in zip(x_vals, y_vals):
        if x == max_x or x == 0:
            adjusted_y_vals.append(y * 10)  # Multiply by 10 for last layer neurons
        else:
            adjusted_y_vals.append(y)
    
    # Update max_y if necessary
    adjusted_max_y = max(adjusted_y_vals)

    # Convert to numpy arrays for plotting
    x_array = np.array(x_vals)
    y_array = np.array(adjusted_y_vals)

    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(x_array, y_array, c=colors, s=100, label='Neurons')

    
    # Set x and y ticks
    plt.xticks(range(max_x + 1))
    if adjusted_max_y < 40:
        plt.yticks(np.arange(1, adjusted_max_y + 1, 1))
    else:
        plt.yticks(np.arange(1, adjusted_max_y + 1, 20))
    
    # Add labels and title
    plt.xlabel("Layer Index")
    plt.ylabel("Neuron Index")

    # Add a legend for communities
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10) 
               for i in range(len(unique_communities))]
    plt.legend(handles, [f"Community {community}" for community in unique_communities], 
               title="Communities", loc='upper right', bbox_to_anchor=(1.15, 1))

    # Save the plot to a file
    plt.savefig(filename, bbox_inches='tight')
    # Save the plot to an EPS file
    plt.savefig(filename.replace(".png", ".eps"), format='eps', bbox_inches='tight')
    plt.close()

def plot_feature_importances(*feature_lists, labels, filename="feature_importance_comparison_vertical.png", top_n=30):
    """
    Plots top N relative feature importances from multiple datasets for comparison in a vertical bar chart.

    Parameters:
    - *feature_lists: Variable number of feature lists, each a list of tuples (feature_name, importance_score).
    - labels: List of labels corresponding to each feature list.
    - filename: Name of the file to save the plot.
    - top_n: Number of top features to plot.
    """

    num_datasets = len(feature_lists)

    # Check if the number of labels matches the number of feature lists
    if len(labels) != num_datasets:
        raise ValueError("The number of labels must match the number of feature lists provided.")

    # Convert each feature list to a dictionary and normalize importance scores
    features_dicts = []
    for i, features in enumerate(feature_lists):
        features_dict = dict(features)
        total_importance = sum(features_dict.values())
        if total_importance == 0:
            total_importance = 1  # Avoid division by zero
        # Normalize the importance scores
        for key in features_dict:
            features_dict[key] = features_dict[key] / total_importance
        features_dicts.append(features_dict)

    # Get the set of all features across all datasets
    feature_names = set()
    for features_dict in features_dicts:
        feature_names.update(features_dict.keys())
    feature_names = sorted(feature_names)

    # Calculate average importance for each feature across all datasets
    avg_importances = []
    for f in feature_names:
        importances = [features_dict.get(f, 0) for features_dict in features_dicts]
        avg = sum(importances) / num_datasets
        avg_importances.append((f, avg))

    # Sort features by average importance descending
    avg_importances.sort(key=lambda x: x[1], reverse=True)
    sorted_feature_names = [f for f, _ in avg_importances[:top_n]]

    # Prepare importances for plotting
    importances_list = []
    for features_dict in features_dicts:
        importances = [features_dict.get(f, 0) * 100 for f in sorted_feature_names]  # Convert to percentages
        importances_list.append(importances)

    # Adjust figure size based on number of features
    fig_width = max(6, top_n * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    x_pos = np.arange(len(sorted_feature_names))
    width = 0.8 / num_datasets  # Total bar width is 0.8, divided by the number of datasets

    # Plot bars for each dataset
    for i, importances in enumerate(importances_list):
        offset = (i - num_datasets / 2) * width + width / 2
        rects = ax.bar(x_pos + offset, importances, width=width, label=labels[i])
        # Add data labels to bars
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # Vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)

    # Add labels, title, etc.
    ax.set_ylabel('Relative Importance (%)')
    ax.set_title('Top {} Relative Feature Importances Comparison'.format(top_n))
    ax.set_xticks(x_pos)
    # Optionally, map feature names to more readable labels
    x_labels = [f.replace('layer_0_neuron_', 'Feature ') for f in sorted_feature_names]
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
    ax.legend()

    # Add grid
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig(filename)
    plt.close()

def compute_community_overlap_matrix(community_lists, list_names=None):
    match_matrix = np.zeros((len(community_lists), len(community_lists)))
    for i in range(len(community_lists[0])):
        # Check if (NAME, community_index) the community_index is for the current i the same
        for d in range(len(community_lists)):
            for e in range(len(community_lists)):
                if community_lists[d][i][1] == community_lists[e][i][1]:
                    match_matrix[d][e] += 1
                else:
                    match_matrix[d][e] += 0
    # Calculate percentage
    match_matrix = match_matrix / len(community_lists[0])
    # Convert to data frame and name columns and row indices based on labels
    match_matrix = pd.DataFrame(match_matrix, columns=list_names, index=list_names)
    return match_matrix

def compare_state_lists(states1, states2):
    """
    Compares two lists of states and returns the percentage of not equal elements.

    Args:
        states1 (list of np.ndarray): The first list of states.
        states2 (list of np.ndarray): The second list of states.

    Returns:
        float: Percentage of states that are not equal between the two lists.
    """
    # Determine the length to compare (up to the shortest list length)
    min_length = min(len(states1), len(states2))

    # Count the number of states that are not equal
    not_equal_count = 0
    for i in range(min_length):
        if not np.array_equal(states1[i], states2[i]):
            not_equal_count += 1

    # Calculate the percentage of not equal states
    percentage_not_equal = (not_equal_count / min_length) * 100

    return percentage_not_equal
