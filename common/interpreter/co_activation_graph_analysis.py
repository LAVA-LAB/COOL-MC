from common.interpreter.interpreter import Interpreter
from common.interpreter.prism_encoder import PrismEncoder
from common.interpreter.co_activation_graph import *



class CoActivationGraphAnalysis(Interpreter):

    def __init__(self, config):
        self.config = config
        self.parse_config(config)

    def parse_config(self, config_str:str) -> None:
        self.name = config_str.split(";")[0]
        self.prop_query1 = config_str.split(";")[1]
        self.prop_query2 = config_str.split(";")[2]

    def interpret(self, env, m_project, model_checking_info):
        # Query1
        mdp_reward_result1, model_checking_info1 = env.storm_bridge.model_checker.induced_markov_chain(m_project.agent, m_project.preprocessors, env, m_project.command_line_arguments['constant_definitions'], self.prop_query1, True)
        #states1 = get_states_from_interconnections(model_checking_info1['state_interconnections'])
        states1 = model_checking_info1['collected_states']
        
       
        
        # Query2
        mdp_reward_result2, model_checking_info2 = env.storm_bridge.model_checker.induced_markov_chain(m_project.agent, m_project.preprocessors, env, m_project.command_line_arguments['constant_definitions'], self.prop_query2, True)
        #states2 = get_states_from_interconnections(model_checking_info2['state_interconnections'])
        states2 =  model_checking_info2['collected_states']
        

        print(compare_state_lists(states1, states2))
        print(len(states1), len(states2))
        # df1_graph - df2_graph
        #df_diff = df1_graph - df2_graph
        # Sum up over all elements
        #sum_diff = df_diff.abs().sum().sum()
        #print(f"Sum of differences between co-activation graphs: {sum_diff}")

        # Create Co-activation graph
        df1_graph = create_coactivation_graph(m_project.agent, states1)
        df2_graph = create_coactivation_graph(m_project.agent, states2)
        # Analyze Communities
        df1_communities, df1_modularity = analyze_community_with_modularity(df1_graph)
        df2_communities, df2_modularity = analyze_community_with_modularity(df2_graph)

        print("Modularity Query1:", df1_modularity, " Modularity Query2:", df2_modularity)
        print(compute_community_overlap_matrix([df1_communities, df2_communities], list_names=[self.prop_query1, self.prop_query2]))

        l1, features1, _ = analyze_centrality2(df1_graph)
        l2, features2, _ = analyze_centrality2(df2_graph)

        # Generate the feature importance plot
        print(features1)
        print(features2)
        print(env.storm_bridge.state_mapper.get_feature_names())

        # Create labels
        label1 = self.prop_query1.replace("Pmax=? [ F ", "").replace("]", "")
        label2 = self.prop_query2.replace("Pmax=? [ F ", "").replace("]", "")

        plot_layer_neuron_importance(l1[0:50], l2[0:50], filename='layer_neuron_importance.png', dataset_names=[label1, label2])
        print(len(states1), len(states2))