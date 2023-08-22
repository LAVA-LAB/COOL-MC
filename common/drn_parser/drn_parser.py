import os
import re
import json
import torch
import numpy as np
def parse_boolean_tuples_from_line(line):
    line = line[3:]
    parts = line.split("&")
    bls = []
    for part in parts:
        if part.startswith("!") and part.find("=") == -1:
            translation_table = str.maketrans("", "", "/[!]")
            part = part.translate(translation_table)
            bls.append((part.strip(), 0))
        elif part.startswith("!") == False and part.find("=") == -1:
            translation_table = str.maketrans("", "", "/[!")
            part = part.translate(translation_table)
            bls.append((part.strip(), 1))
    return bls

class State:

    def __init__(self, state_id, state_id_line):
        self.state_id = state_id
        self.state_id_line = state_id_line
        self.state_valuation = ""
        self.features = {}
        self.action_transitions = {}

    def create_state_valuation(self, state_valuation):
        """
        Creates a feature assignment from the state valuation.
        """
        self.state_valuation = state_valuation
        feature_pattern = re.compile(r"(\w+)\s*=\s*(!?\w+)")
        feature_boolean_pattern = re.compile(r"(!?\w+)(?=\s*&)")
        feature_matches = feature_pattern.findall(self.state_valuation)
        feature_boolean_matches = feature_boolean_pattern.findall(self.state_valuation)
        features = {}
        for feature_match in feature_matches:
            features[feature_match[0]] = int(feature_match[1])
        bls = parse_boolean_tuples_from_line(self.state_valuation)
        for bl in bls:
            #print(bl)
            features[bl[0]] = bl[1]
        self.features = features
        #print(self.features)

    def get_numpy_state(self, storm_bridge):
        """
        Returns the numpy state.
        """
        return storm_bridge.parse_state(json.dumps(self.features))

    def add_action_transitions(self, action_name, action_transitions):
        self.action_transitions[action_name] = action_transitions

    def __str__(self):
        """
        Returns a string representation of the state.
        """
        return "State " + str(self.state_id) + " with state valuation " + self.state_valuation + " and features " + str(self.features) + " and action transitions " + str(self.action_transitions)

    def write_to_file(self, file, label, action_names, action_idizes, action_probs):
        if label == True:
            action_prob_mapper = {}
            next_state_probs = {}
            for i in range(len(action_idizes)):
                action_prob_mapper[action_names[i]] = action_probs[i].double()
            print(self.state_id)
            print(action_prob_mapper)
            #exit(0)
            file.write(self.state_id_line)
            file.write(self.state_valuation)
            #print(self.action_transitions)
            file.write("\taction ACTION_MERGED\n")
            for action_name in action_names:
                # if action is not available
                if action_name not in self.action_transitions.keys():
                    raise Exception("Action not available: ", action_name)
                # Update each transition prob
                for state_id in self.action_transitions[action_name].keys():
                    #action_prob_mapper[action_names[i]] *= self.action_transitions[action_name][state_id]
                    if state_id not in next_state_probs.keys():
                        next_state_probs[state_id] = torch.tensor(self.action_transitions[action_name][state_id],dtype=torch.float64) * action_prob_mapper[action_name]
                    else:
                        next_state_probs[state_id] += torch.tensor(self.action_transitions[action_name][state_id],dtype=torch.float64) * action_prob_mapper[action_name]
                    #file.write(f"\t\t{state_id} : {self.action_transitions[action_name][state_id] * action_prob_mapper[action_name]}\n")


            # Check if tensor is 1
            if torch.all(s == 1)==False:
                if s < 1:
                    next_state_probs[key] -= (1-s)
                else:
                    next_state_probs[key] += (1-s)


            for idx, state_id in enumerate(next_state_probs.keys()):
                print("State id: ", state_id, " and prob: ", next_state_probs[state_id].item())
                file.write(f"\t\t{state_id} : {next_state_probs[state_id].item()}\n")


        else:
            file.write(self.state_id_line)
            file.write(self.state_valuation)
            file.write("\taction __NOLABEL__")
            file.write(f"\n\t\t{self.state_id} : 1")


class DRNParser():

    def __init__(self, file_path, env, agent):
        self.file_path = file_path
        self.env = env
        self.storm_bridge = env.storm_bridge
        self.state_mapper = env.storm_bridge.state_mapper
        self.agent = agent
        if not os.path.exists(self.file_path):
            raise Exception("File does not exist: ", self.file_path)
        self.parse()

    def get_action_names(self, action_idizes):
        all_action_names = []
        for action_index in action_idizes:
            action_name = self.env.action_mapper.actions[action_index]
            all_action_names.append(action_name)
        return all_action_names

    def parse(self):
        """
        Parses the file in the file path.
        At the same time, it writes a modified version into another file.

        """
        current_state = None
        action_name = None
        action_transitions = {}
        with open("new_file.drn", "w") as write_to_file:
            with open(self.file_path, "r") as read_from_file:
                state_valuation_str = ""
                for line in read_from_file:
                    #print(line)
                    #write_to_file.write(line)
                    if line.startswith("state") and line.strip()!="":
                        if current_state != None:
                            current_line = line.strip()
                            current_state.add_action_transitions(action_name, action_transitions)
                            print("=====================================")
                            #print(current_state)
                            action_transitions = {}
                            action_name = None
                            if '__NOLABEL__' not in current_state.action_transitions.keys():
                                # TODO: Modify state-transitions
                                # TODO: Write new state-transitions to file
                                print("MODIFY")
                                state = current_state.get_numpy_state(self.storm_bridge)
                                action_idizes, action_probs = self.agent.model_checking_select_action(state)
                                action_names = self.get_action_names(action_idizes)
                                print(self.agent.model_checking_select_action(state))
                                print(self.get_action_names(action_idizes))
                                current_state.write_to_file(write_to_file, True, action_names, action_idizes, action_probs)
                            else:
                                current_state.write_to_file(write_to_file, False, action_names, action_idizes, action_probs)
                                # For each action transition, create a new transition in MERGED_ACTION
                                # Something like this:
                                # MERGED_ACTION:
                                #   0: 0.5 * original action A transition1 // 0.5 is the probability of the action A
                                #   0: 0.5 * original action A transition2 // 0.5 is the probability of the action A
                                #   0: 0.5 * original action B transition1 // 0.5 is the probability of the action B
                                #   0: 0.5 * original action B transition2 // 0.5 is the probability of the action B

                        #print(line)
                        state_id = int(line.split(" ")[1])

                        current_state = State(state_id, line)
                        state_valuation_str = ""
                    elif line.startswith("//") or line.strip().endswith("]"):
                        state_valuation_str += line
                        if line.strip().endswith("]"):
                            current_state.create_state_valuation(state_valuation_str)
                    elif line.strip().startswith("action"):
                        current_line = line.strip()
                        #print(current_line)
                        if len(action_transitions.keys()) != 0:
                            #print(action_name + " " + str(action_transitions))
                            current_state.add_action_transitions(action_name, action_transitions)

                        action_name = current_line.split(" ")[1]
                        action_transitions = {}
                    elif line.strip().find(":") != -1 and action_name != None:
                        current_line = line.strip()
                        #print(current_line)
                        action_transitions[int(current_line.split(":")[0].strip())] = float(current_line.split(":")[1].strip())
                    if current_state == None:
                        write_to_file.write(line)
            print("=====================================")
            current_state.add_action_transitions(action_name, action_transitions)
            print(current_state)
            if '__NOLABEL__' not in current_state.action_transitions.keys():
                # TODO: Modify state-transitions
                # TODO: Write new state-transitions to file
                print("MODIFY")
                state = current_state.get_numpy_state(self.storm_bridge)
                action_idizes, action_probs = self.agent.model_checking_select_action(state)
                action_names = self.get_action_names(action_idizes)
                print(self.agent.model_checking_select_action(state))
                print(self.get_action_names(action_idizes))
                current_state.write_to_file(write_to_file, True, action_names, action_idizes, action_probs)
            else:
                current_state.write_to_file(write_to_file, False, action_names, action_idizes,  action_probs)











