import ast
import numpy as np

class State:

    def __init__(self, id, state_line=None) -> None:
        self.id = id
        self.state_line = state_line
        self.actions = {}
        self.last_action = None
        self.line_in_between_state_id_and_actions = []
        self.terminal_state = False

    def add_lines_in_between_state_id_and_actions(self, line):
        if self.last_action is None and line.startswith("\taction ") == False:
            self.line_in_between_state_id_and_actions.append(line)

    def add_action(self, action_line):
        if action_line.startswith("\taction "):
            self.last_action = action_line.split(" ")[-1].strip()
            self.actions[self.last_action] = {}
            if self.last_action == "__NOLABEL__":
                self.terminal_state = True


    def add_action_probability(self, action_line):
        # Parse: 		9 : 0.333
        if action_line.find(":")!=-1:
            action, probability = action_line.split(":")
            next_state_id = int(action.strip())
            probability = float(probability.strip())
            if self.last_action is not None:
                self.actions[self.last_action][next_state_id] = probability

    def get_all_next_states(self):
        for action in self.actions:
            for next_state_id in self.actions[action]:
                yield next_state_id

    def is_self_looping(self):
        # Return only true if all transitions are self-loops
        for action in self.actions:
            for next_state_id in self.actions[action]:
                if next_state_id != self.id:
                    return False
        return True

    def get_state_update(self, env, agent, id_state_mapper):
        all_next_state_ids = list(self.get_all_next_states())
        all_next_state_ids = list(set(all_next_state_ids))
        n_state = State(self.id)
        n_state.state_line = self.state_line
        n_state.line_in_between_state_id_and_actions = self.line_in_between_state_id_and_actions
        n_state.actions['meta_action'] = {}
        if self.id not in id_state_mapper:# or self.is_self_looping():
            n_state.actions = {}
            n_state.actions['__NOLABEL__'] = {self.id: 1}
            n_state.terminal_state = True
            return n_state
        for next_state_id in all_next_state_ids:
            prob_sum = 0
            for action in self.actions:
                if action == "__NOLABEL__":
                    return self
                if next_state_id in self.actions[action]:
                    prob_sum += self.actions[action][next_state_id] * agent.get_action_name_probability(env, action, id_state_mapper[self.id])
                n_state.actions['meta_action'][next_state_id] = prob_sum
        return n_state

    def __str__(self) -> str:
        important_label = ""
        if self.terminal_state == True and self.state_line is not None:
            important_label = "asdfasdfa"


        if self.state_line is not None:
            str_block = self.state_line + " " + important_label + "\n"
        else:
            str_block = "state " + str(self.id) + " " + "\n"
        for line in self.line_in_between_state_id_and_actions:
            str_block += line
        for action in self.actions:
            str_block += "\taction " + action + "\n"
            for next_state_id in self.actions[action]:
                str_block += "\t\t" + str(next_state_id) + " : " + str(self.actions[action][next_state_id]) + "\n"
        return str_block




class StochasticPolicyChecker:

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.id_state_mapper = {}

    def create_id_state_mapper(self, id_state_mapping_file_path):
        with open(id_state_mapping_file_path, 'r') as file:
                # Read the file line by line
                for line in file:
                    state_id, state_str = line.split(";")
                    state_id = int(state_id)
                    state = np.array(ast.literal_eval(state_str))
                    self.id_state_mapper[state_id] = state



    def update_drn_file(self, drn_file_path):
        # Read drn file
        # Extract the states line by line and store the line number of the state starting line
        # For each state, create a new action called meta_action

        with open("new_file.drn", 'w') as new_file:
            with open(drn_file_path, 'r') as file:
                number_of_states = -1
                state = None
                for line in file:
                        if line.startswith("state"):
                            if state!=None:
                                n_state = state.get_state_update(self.env, self.agent, self.id_state_mapper)
                                new_file.write(n_state.__str__())

                            state_id = int(line.split(" ")[1])

                            if len(line.split(" "))>2:
                                state_line = str(line).strip()
                                state = State(state_id,state_line=state_line)
                            else:
                                state_line = None
                                state = State(state_id,state_line=state_line)

                        elif state is not None:
                            state.add_lines_in_between_state_id_and_actions(line)
                            state.add_action(line)
                            state.add_action_probability(line)
                        if state == None:
                            # Check if line starts with a number
                            if number_of_states != -1 and line.strip().isdigit():
                                line = str(number_of_states) + "\n"
                            elif line.strip().isdigit():
                                number_of_states = int(line.strip())

                            new_file.write(line)

                n_state = state.get_state_update(self.env, self.agent, self.id_state_mapper)
                new_file.write(n_state.__str__())









