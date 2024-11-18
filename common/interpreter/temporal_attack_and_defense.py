from common.interpreter.interpreter import Interpreter
from common.interpreter.prism_encoder import PrismEncoder
from common.preprocessors.single_agent_fgsm import FGSM
from common.preprocessors.single_agent_deepfool_attack import DeepFool
import stormpy
import random
class TemporalAdvAttackAndDefense:

    def __init__(self, config):
        self.config = config
        self.parse_config(config)

    def parse_config(self, config_str:str) -> None:
        self.name = config_str.split(";")[0]
        self.prop_query = config_str.split(";")[1]
        self.epsilon = float(config_str.split(";")[2])
        self.attack_name = config_str.split(";")[3]
        self.defense_name = config_str.split(";")[4]

    def attack_successful(self, env, rl_agent, state):
        current_flag = 0
        attack_config_str = "fgsm;"+str(self.epsilon)
        original_action = rl_agent.select_action(state, True)
        if self.attack_name == "fgsm":
            attacker = FGSM(env.storm_bridge.state_mapper, attack_config_str, "fgsm")
        elif self.attack_name == "deepfool":
            attacker = DeepFool(env.storm_bridge.state_mapper, f"deepfool;{self.epsilon};10", "deepfool") # epsilon = overshoot
        else:
            raise Exception("Attack not supported")
        adv_state = attacker.preprocess(rl_agent, state, env.action_mapper, "", True)
        adv_action = rl_agent.select_action(adv_state, True)
        if original_action != adv_action:
            current_flag = 1
        else:
            current_flag = 0
        # floor
        if self.defense_name == "floor":
            for i in range(0, len(adv_state)):
                adv_state[i] = int(adv_state[i])
        elif self.defense_name == "round":
            for i in range(0, len(adv_state)):
                adv_state[i] = round(adv_state[i])
        else:
            raise Exception("Defense not supported")
        def_action = rl_agent.select_action(adv_state, True)
        if original_action != def_action:
            current_flag += 3
        else:
            current_flag += 2
        return current_flag
        

    def interpret(self, env, m_project, model_checking_info):
        n_interconnections = []
        # np_state, action_idx, action_name, all_next_states, env.storm_bridge.state_mapper.get_feature_names()
        for idx, elem in enumerate(model_checking_info["state_interconnections"]):
            state = elem[0]
            action_idx = elem[1]
            action_name = elem[2]
            all_next_states = elem[3]
            feature_names = elem[4]
            # ATTACK STATE
            state_importance = self.attack_successful(env, m_project.agent, state)
            next_state_importances = []
            n_feature_value = 0
            for next_state in all_next_states:
                # ATTACK NEXT STATE
                n_feature_value = self.attack_successful(env, m_project.agent, next_state[0])
                next_state_importances.append(n_feature_value)

            n_feature_value = 0
            n_feature_value = state_importance
            n_interconnections.append((state, action_idx, action_name, all_next_states, feature_names, n_feature_value, next_state_importances))
            

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