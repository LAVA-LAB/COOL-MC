from common.interpreter.interpreter import Interpreter
from common.interpreter.prism_encoder import PrismEncoder
from common.interpreter.co_activation_graph import *
import torch
import torch.nn.functional as F
import stormpy
import torch
import numpy as np
import pandas as pd


class CoActivationGraphAnalysisCriticalStates(Interpreter):

    def __init__(self, config):
        self.config = config
        self.parse_config(config)

    def parse_config(self, config_str:str) -> None:
        self.name = config_str.split(";")[0]
        self.prop_query1 = config_str.split(";")[1]
        self.threshold = float(config_str.split(";")[2])

    def interpret(self, env, m_project, model_checking_info):
        # Query1
        mdp_reward_result1, model_checking_info1 = env.storm_bridge.model_checker.induced_markov_chain(m_project.agent, m_project.preprocessors, env, m_project.command_line_arguments['constant_definitions'], self.prop_query1, True)
        #states1 = get_states_from_interconnections(model_checking_info1['state_interconnections'])
        states1 = model_checking_info1['collected_states']
        criticality = []
        critical_states = []
        not_critical_states = []
        for state in states1:
            next_state_importance = float(m_project.agent.q_eval.forward(state).max().item()-m_project.agent.q_eval.forward(state).min().item())
            print(next_state_importance)
            if next_state_importance>=self.threshold:
                #print("Critical state", state.shape)
                critical_states.append(state)
            else:
                #print("Not critical state", state.shape)
                not_critical_states.append(state)

        # Create Co-activation graph
        df1_graph = create_coactivation_graph(m_project.agent, critical_states)
        df2_graph = create_coactivation_graph(m_project.agent, not_critical_states)
        # Analyze Communities
        df1_communities, df1_modularity = analyze_community_with_modularity(df1_graph)
        df2_communities, df2_modularity = analyze_community_with_modularity(df2_graph)

        print("Modularity Query1:", df1_modularity, " Modularity Query2:", df2_modularity)
        print(compute_community_overlap_matrix([df1_communities, df2_communities], list_names=["Critical", "Not Critical"]))

        l1, features1, _ = analyze_centrality2(df1_graph)
        l2, features2, _ = analyze_centrality2(df2_graph)

        # Generate the feature importance plot
        print(features1)
        print(features2)
        print(env.storm_bridge.state_mapper.get_feature_names())

        plot_layer_neuron_importance(l1[0:50], l2[0:50], filename='layer_neuron_importance_critical.png', dataset_names=["Critical", "Not Critical"])
        print(len(critical_states), len(not_critical_states))
       
        
        