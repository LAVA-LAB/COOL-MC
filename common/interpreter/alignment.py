import matplotlib.pyplot as plt
import numpy as np
from common.preprocessors.change_action_of_state import *
from common.interpreter.interpreter import *

class AlginmentInterpreter(Interpreter):

    def __init__(self, config):
        self.config = config
        self.action_values = {}

    def assign_value_to_action(self, action_name, original_result, current_result, collected_state):
        if original_result != current_result:
            print("=====================================")
            print("collected_state: ", collected_state)
            print("action_name: ", action_name)
            print("Original result: ", original_result)
            print("Current result: ", current_result)
        if action_name not in self.action_values:
            self.action_values[action_name] = original_result - current_result
        else:
            self.action_values[action_name] += original_result - current_result


    def interpret(self, env, m_project, model_checking_info):
        original_result = model_checking_info['mdp_result']
        action_labels = env.action_mapper.actions
        action_type_values = {}
        counter = 0
        # Alignment
        # Init reachability probability matrix
        # For each collected state
        print("Total Iterations:", len(model_checking_info['collected_states'])*len(action_labels))
        for idx, collected_state in enumerate(model_checking_info['collected_states']):
            collected_action_idx = model_checking_info['collected_action_idizes'][idx]
            collected_action_name = action_labels[collected_action_idx]
            for action_idx, action in enumerate(action_labels):
                counter += 1
                if action_idx == collected_action_idx:
                    continue
                else:
                    preprocessor_str = "change_action_of_state;" + np.array2string(collected_state) + ";" + str(action_idx)
                    preprocessor = ChangeActionOfState(env.storm_bridge.state_mapper, preprocessor_str, "inner_loop")
                    m_project.preprocessors = [preprocessor]
                    # create preprocessor that takes collected state and the wanted action as input
                    mdp_reward_result, _ = env.storm_bridge.model_checker.induced_markov_chain(m_project.agent, m_project.preprocessors, env, m_project.command_line_arguments['constant_definitions'], m_project.command_line_arguments['prop'], None)
                    self.assign_value_to_action(collected_action_name, original_result, mdp_reward_result, collected_state)
        print(self.action_values)









        ## For each action != selected action, we want to know the safety measurement outcome
        ## Base on this outcome, classify the action and store in a dictionary d[state][action] = CLASS_LABEL
        ## Based on the induced DTMC, calculate the reachability probability to every state from init state
        ## Add weighted reachability probabilities with respect to the Q(s,action) to probability matrix
        # Plot action class distribution
        # Calculate average reachability probability
        # Plot average reachability probability as heatmap
        # Intense highlighted rows, indicate states that are likely to be from interest for the agent.


