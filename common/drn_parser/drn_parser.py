import re

def parse_drn(drn_file):
    state_regex = r'^state (\d+).*'
    action_started = False
    first_state = True
    current_state = {}
    all_states = []
    with open(drn_file, 'r') as file:
        for line in file.readlines():
            match = re.search(state_regex, line)
            if match:
                state = int(match.group(1))
                action_started = False
                if first_state:
                    current_state = {"state_id": state, "transitions": [], "transition_probs" : [], "init" : first_state}
                    first_state = False
                else:
                    all_states.append(current_state)
                    current_state = {"state_id": state, "transitions": [], "transition_probs" : [], "init" : first_state}
            else:
                if line.find('action')!=-1:
                    action_started = True
                elif action_started:
                    t_state_id = line.split(':')[0].strip()
                    current_state["transitions"].append(int(t_state_id))
                    t_prob = line.split(':')[1].strip()
                    current_state["transition_probs"].append(float(t_prob))
    all_states.append(current_state)

    nodes = []
    for state in all_states:
        nodes.append(Node(state["state_id"], state["transitions"], state["transition_probs"], state["init"]))

    return nodes


class Node:

    def __init__(self, state_id, transitions, transition_probs, init):
        self.state_id = state_id
        self.neighbours = []
        self.neighbours = {transitions[i]: transition_probs[i] for i in range(len(transitions))}
        self.init = init
        self.reachability = 0

    def __str__(self):
        if self.init:
            return f"x_{self.state_id} = 1"
        else:
            eq_str = f"x_{self.state_id} = "
            for neighbour in self.neighbours:
                eq_str += f"{self.neighbours[neighbour]}x_{neighbour} + "
            return eq_str[:-3]








drn_file = 'test.drn'
nodes = parse_drn(drn_file)
#nodes = reversed(nodes)

for node in nodes:
    print(f"{node}")
