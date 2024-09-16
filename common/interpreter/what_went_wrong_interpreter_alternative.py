


class WhatWentWrongInterpreterAlternative:

    def __init__(self, config):
        self.config = config
        self.parse_config(config)

    def parse_config(self, config_str:str) -> None:
        self.name = config_str.split(";")[0]
        self.condition_type = config_str.split(";")[1]


    def next_state_condition(self, state, next_state, feature_names, action, prob):
        # Get feature name index
        #print("HERE", self.condition_type)
        if self.condition_type == "P=? [ F energy=0 ]":
            #print(next_state, "HERE")
            targeted_feature_name = "energy"
            targeted_feature_index = feature_names.index(targeted_feature_name)
            #print(next_state[targeted_feature_index])
            if next_state[targeted_feature_index] == 0 and state[targeted_feature_index]>0:
                #exit(0)
                return True
            else:
                return False
        elif self.condition_type == "P=? [ F dirt1=-5 ]":
            targeted_feature_name = "dirt1"
            targeted_feature_index = feature_names.index(targeted_feature_name)
            
            if next_state[targeted_feature_index] == -5:
                return True
            else:
                return False
        elif self.condition_type == "P=? [ F dirt1=-2]":
            targeted_feature_name = "dirt1"
            targeted_feature_index = feature_names.index(targeted_feature_name)
            
            if next_state[targeted_feature_index] == -2:
                return True
            else:
                return False
        elif self.condition_type == "P=? [ F dirt1=-4]":
            targeted_feature_name = "dirt1"
            targeted_feature_index = feature_names.index(targeted_feature_name)
            
            if next_state[targeted_feature_index] == -4:
                return True
            else:
                return False
        else:
            raise ValueError("Condition type not supported")
        
        

    def interpret(self, env, m_project, model_checking_info):
        wrong_action_selection = 0
        wrong_parsing = 0
        #print(env.storm_bridge.prism_file_content)
        original_mdp_result = float(model_checking_info['mdp_reward_result'])
        n_interconnections = []
        state_filters = []
        counter = 0
        error_list = []
        for idx, elem in enumerate(model_checking_info["state_interconnections"]):
            state = elem[0]
            if all(x >= 0 for x in state):
                for next_state, prob in elem[3]:
                    
                    # Check if any element in the next state is negative
                    #print(next_state)
                    if self.next_state_condition(state, next_state, elem[-1], elem[2], prob):
                        print('='*50)
                        print("Something went wrong with", prob, "likelihood in the state: ", elem[0], " with action: ", elem[2], " because", next_state)
                        action_index = m_project.agent.get_nt_action_alternative(state, 1)
                        
                        if action_index is None:
                            action_index = 0
                            wrong_action_selection+=1
                        state_filters.append((elem[0], action_index))
                        counter+=1
                    break


            
        # Add to agent
        m_project.agent.state_filters = state_filters
        # Model Check again
        mdp_reward_result, model_checking_info = env.storm_bridge.model_checker.induced_markov_chain(m_project.agent, m_project.preprocessors, env, m_project.command_line_arguments['constant_definitions'], m_project.command_line_arguments['prop'], False)
        print("MDP Result before filtering:", original_mdp_result)
        print("MDP Reward Result after filtering:", mdp_reward_result)
        print("Wrong action selection:", wrong_action_selection)
        print("Wrong parsing:", wrong_parsing)

        # Write word list line by line to file
        with open("error_types1.txt", "w") as f:
            for error in error_list:
                f.write(error + "\n")