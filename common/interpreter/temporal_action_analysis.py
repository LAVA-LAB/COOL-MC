from common.interpreter.interpreter import Interpreter
from common.interpreter.prism_encoder import PrismEncoder
import stormpy
import random
class TemporalActionAnalysis:

    def __init__(self, config):
        self.config = config
        self.parse_config(config)

    def parse_config(self, config_str:str) -> None:
        self.name = config_str.split(";")[0]
        self.prop_query = config_str.split(";")[1]
        

    def interpret(self, env, m_project, model_checking_info):
        # action name to action index
        #reference_action_idx = env.action_mapper.action_name_to_action_index(self.action_name)
        n_interconnections = []
        # np_state, action_idx, action_name, all_next_states, env.storm_bridge.state_mapper.get_feature_names()
        for idx, elem in enumerate(model_checking_info["state_interconnections"]):
            state = elem[0]
            action_idx = elem[1]
            action_name = elem[2]
            all_next_states = elem[3]
            feature_names = elem[4]
            
            current_action_index = m_project.agent.select_action(state, True)
            next_state_action_indizes = []
            n_feature_value = None
            for next_state in all_next_states:
                next_action_index = m_project.agent.select_action(next_state[0],True)
                #print(next_action_index)
                n_feature_value = next_action_index
                next_state_action_indizes.append(n_feature_value)

         
            n_feature_value = current_action_index
            n_interconnections.append((state, action_idx, action_name, all_next_states, feature_names, n_feature_value, next_state_action_indizes))
            
        trans_counter = 0
        for interconnection in n_interconnections:
            for next_state in interconnection[3]:
                trans_counter += 1
       
        prism_encoder = PrismEncoder(n_interconnections)
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
        print("Action Index Order (starting with 0):", env.action_mapper.actions)