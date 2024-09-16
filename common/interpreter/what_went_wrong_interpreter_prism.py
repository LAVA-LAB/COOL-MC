import re
import json
from openai import OpenAI
def generate_gpt_text(client, system_text, input_text):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": input_text}
        ],
        temperature=1,
        max_tokens=4096,
        top_p=0,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content



def state_feature_zipper(state, features):
    return "[" + ', '.join([f'{feature}: {value}' for feature, value in zip(features, state)]) + "]"



def extract_action(json_string):
    # Use regex to find the JSON object
    json_match = re.search(r'(\{.*?"action"\s*:\s*"IDLE".*?\})', json_string)
    if json_match:
        json_text = json_match.group(1)
        try:
            result_dict = json.loads(json_text)
            return result_dict
        except json.JSONDecodeError:
            print("Failed to decode JSON from the response.")
            return None
    else:
        print("No matching JSON found in the response.")
        return None


def extract_json_objects(json_string):
    # Use regex to find all JSON objects in the string
    json_matches = re.findall(r'\{.*?\}', json_string, re.DOTALL)
    json_objects = []

    for json_text in json_matches:
        try:
            result_dict = json.loads(json_text)
            json_objects.append(result_dict)
        except json.JSONDecodeError:
            print(f"Failed to decode JSON: {json_text}")

    return json_objects[0]


class WhatWentWrongInterpreterPRISM:

    def __init__(self, config):
        self.config = config
        self.parse_config(config)

    def parse_config(self, config_str:str) -> None:
        self.name = config_str.split(";")[0]
        self.condition_type = config_str.split(";")[1]


    def get_llm_response(self, prism_file_content, state, action, prob, feature_names, next_state):
        print(state, feature_names)
        model = OpenAI(api_key="OPENAI_API_KEY")
        system_text = "The reinforcement learning environment is modeled as a PRISM model:\n" + prism_file_content
        #input_text = "We are interested in avoiding the bad behavior described as a PCTL formula: " + self.condition_type + "."
        input_text = f"What went wrong with likelihood {prob} in the state " + state_feature_zipper(state, feature_names) + " with action " + str(action) + f" ending up in {state_feature_zipper(next_state, feature_names)}? Explain it to me.\n"
        input_text += "Return me additionally in JSON format an valid alternative action and the type of error {\"action\": \"ACTION_NAME\", \"error\":\"ERROR_TYPE\"}."
        input_text += "Be aware of the energy consumptions and all sources of potential mistakes. "
        input_text += "The error categories are as follows: out_of_energy, collision, charging_error, other. "
        input_text += "Note that a negative dirt1 value indicates only indicates that the environment has terminated (nothing else)."
        print(input_text)
        response = generate_gpt_text(model, system_text, input_text)
        print(response)
        return extract_json_objects(response)["action"], extract_json_objects(response)["error"], response, input_text
        
        
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
        all_responses = []
        for idx, elem in enumerate(model_checking_info["state_interconnections"]):
            state = elem[0]
            if all(x >= 0 for x in state):
                for next_state, prob in elem[3]:
                    # Check if any element in the next state is negative
                    #print(next_state)
                    if self.next_state_condition(state, next_state, elem[-1], elem[2], prob):
                        print('='*50)
                        print("Something went wrong with", prob, "likelihood in the state: ", elem[0], " with action: ", elem[2], " because", next_state)
                        # Query the LLM for alternative action
                        try:
                            action_name, error, response, input_text = self.get_llm_response(env.storm_bridge.prism_file_content, elem[0], elem[2], prob, elem[-1], next_state)
                            all_responses.append(input_text + "\n" + response)
                            all_responses.append("====================================")
                            #action_name = "CLEAN21"
                            print(action_name, error)
                            error_list.append(error)
                        except Exception as e:
                            print(e)
                            action_name = "safaasfadsfadsfadsfdsafas"
                            wrong_parsing+=1
                        action_index= env.action_mapper.action_name_to_action_index(action_name)
                        if action_index is None:
                            action_index = 0
                            wrong_action_selection+=1
                        state_filters.append((elem[0], action_index))
                        counter+=1
                    break

            #if counter > 20:
            #break

            
        # Add to agent
        m_project.agent.state_filters = state_filters
        # Model Check again
        mdp_reward_result, model_checking_info = env.storm_bridge.model_checker.induced_markov_chain(m_project.agent, m_project.preprocessors, env, m_project.command_line_arguments['constant_definitions'], m_project.command_line_arguments['prop'], False)
        print("MDP Result before filtering:", original_mdp_result)
        print("MDP Reward Result after filtering:", mdp_reward_result)
        print("Wrong action selection:", wrong_action_selection)
        print("Wrong parsing:", wrong_parsing)
        print("Total number of state_filters:", len(m_project.agent.state_filters))

        # Write word list line by line to file
        import time
        current_time = str(int(time.time()))
        with open(f"error_types{current_time}.txt", "w") as f:
            for error in error_list:
                f.write(error + "\n")

        # Write responses to file
        with open(f"responses{current_time}.txt", "w") as f:
            for response in all_responses:
                f.write(response + "\n")