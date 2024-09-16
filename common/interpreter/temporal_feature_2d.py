from common.interpreter.interpreter import Interpreter
from common.interpreter.prism_encoder import PrismEncoder
from common.interpreter.prism_encoder_2d import PrismEncoder2D
import torch
import torch.nn.functional as F
import stormpy

def rank_state_features_via_saliency(model, input_tensor, chosen_action):
    """
    Ranks state features via saliency maps for the currently chosen action.

    Args:
    model (torch.nn.Module): The PyTorch neural network model.
    input_tensor (torch.Tensor): The input tensor representing the state.
    chosen_action (int): The index of the chosen action.

    Returns:
    List[Tuple[int, float]]: A list of tuples where each tuple contains the index of the feature and its saliency value, sorted by saliency.
    """
    
    # Set the model in evaluation mode
    model.eval()
    
    # Ensure the input requires gradient
    input_tensor.requires_grad = True
    
    # Forward pass to get the output
    output = model(input_tensor)
    
    # Zero out gradients
    model.zero_grad()
    
    # Get the output for the chosen action
    chosen_output = output[0, chosen_action]
    
    # Backward pass to compute gradients
    chosen_output.backward()
    
    # Get the gradients of the input
    gradients = input_tensor.grad.data
    
    # Calculate the saliency map by taking the absolute value of the gradients
    saliency = gradients.abs().squeeze()
    
    # Rank the features based on the saliency values
    saliency_values = saliency.flatten()
    ranked_features = [(i, val.item()) for i, val in enumerate(saliency_values)]
    ranked_features.sort(key=lambda x: x[1], reverse=True)
    
    return ranked_features

class TemporalFeature2D(Interpreter):

    def __init__(self, config):
        self.config = config
        self.parse_config(config)

    def parse_config(self, config_str:str) -> None:
        self.name = config_str.split(";")[0]
        self.prop_query = config_str.split(";")[1]
        self.target_feature_name = config_str.split(";")[2]
        self.treshold = float(config_str.split(";")[3])

    def interpret(self, env, m_project, model_checking_info):
        n_interconnections = []
        # np_state, action_idx, action_name, all_next_states, env.storm_bridge.state_mapper.get_feature_names()
        for _, elem in enumerate(model_checking_info["state_interconnections"]):
            state = elem[0]
            action_idx = elem[1]
            action_name = elem[2]
            all_next_states = elem[3]
            feature_names = elem[4]
            
            # Get current feature rank
            tensor_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            ranked_features = rank_state_features_via_saliency(m_project.agent.q_eval, tensor_state, action_idx)
            target_index = feature_names.index(self.target_feature_name)
            current_feature_rank = next((i for i, (idx, _) in enumerate(ranked_features) if idx == target_index), None)
            
            # Get next feature ranks
            feature_ranks = []
            for next_state in all_next_states:
                #print(next_state)
                next_tensor_state = torch.tensor(next_state[0], dtype=torch.float32).unsqueeze(0)
                n_action_idx = m_project.agent.q_eval(next_tensor_state).argmax().item()
                ranked_features = rank_state_features_via_saliency(m_project.agent.q_eval, next_tensor_state, n_action_idx)
                next_target_index = feature_names.index(self.target_feature_name)
                # Get the rank of the target feature
                next_feature_rank = next((i for i, (idx, _) in enumerate(ranked_features) if idx == next_target_index), None)

                feature_ranks.append(next_feature_rank)

            # Get current state importance
            state_importance = float(m_project.agent.q_eval.forward(state).max().item()-m_project.agent.q_eval.forward(state).min().item())
            next_state_importances = []
            n_feature_value = 0
            for next_state in all_next_states:
                next_state_importance = float(m_project.agent.q_eval.forward(next_state[0]).max().item()-m_project.agent.q_eval.forward(next_state[0]).min().item())
                if next_state_importance>=self.treshold:
                    n_feature_value = 1
                else:
                    n_feature_value = 0
                next_state_importances.append(n_feature_value)

            n_feature_value = 0
            if state_importance>=self.treshold:
                n_feature_value = 1
            else:
                n_feature_value = 0
            
            # Append the interconnection
            n_interconnections.append((state, action_idx, action_name, all_next_states, feature_names, current_feature_rank, feature_ranks, n_feature_value, next_state_importances))
            

        prism_encoder = PrismEncoder2D(n_interconnections)
        prism_encoder.encode()
        prism_encoder.to_file(self.name + ".prism")


        prism_program = stormpy.parse_prism_program(self.name + ".prism")
        model = stormpy.build_model(prism_program)
        
        properties = stormpy.parse_properties(self.prop_query, prism_program)
        model = stormpy.build_model(prism_program, properties)
        result = stormpy.model_checking(model, properties[0])
        initial_state = model.initial_states[0]
        stormpy.export_to_drn(model,"model.drn")
        print("====================================")
        print(self.name)
        print("Number of states: {}".format(model.nr_states))
        print("Number of transitions: {}".format(model.nr_transitions))
        print("Property Query:", self.prop_query)
        print("Temporal Explainability Result:",result.at(initial_state))