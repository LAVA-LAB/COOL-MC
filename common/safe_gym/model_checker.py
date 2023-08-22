"""
This module provides the ModelChecker for the RL model checking.
"""
import json
import time
from typing import Tuple
import numpy as np
import stormpy
from stormpy.utility.utility import JsonContainerRational
from stormpy.storage.storage import SimpleValuation
import common
from common.safe_gym.state_mapper import StateMapper
from common.rl_agents.agent import *

class State:

    def __init__(self, id, json_str) -> None:
        self.id = id
        self.json_str = str(json_str)
        self.actions = {}

    def add_action(self, action_name):
        self.last_action = action_name
        self.actions[self.last_action] = {}


    def add_action_probability(self, action, probability, next_state_id):
        self.actions[self.last_action][next_state_id] = probability

    def get_all_next_states(self):
        for action in self.actions:
            for next_state_id in self.actions[action]:
                yield next_state_id

    def get_prob_for_next_state_of_action(self, action, next_state_id):
        try:
            return self.actions[action][next_state_id]
        except:
            return 0


    def get_state_update(self, env, agent):
        all_next_state_ids = list(self.get_all_next_states())
        all_next_state_ids = list(set(all_next_state_ids))
        n_state = State(self.id, self.json_str)
        n_state.actions['meta_action'] = {}
        total_sum = 0
        #print("=========")
        #print("State:",self.id)
        # Iterate over all state ids
        for next_state_id in all_next_state_ids:
            # For each action
            next_state_prob = 0
            for action in self.actions:
                # if next state id is in the actions
                if next_state_id in self.actions[action]:
                    #print("Transition Prob", self.actions[action][next_state_id])
                    #print("Action Prob", agent.get_action_name_probability(env, action, env.storm_bridge.parse_state(self.json_str)))
                    next_state_prob += self.actions[action][next_state_id] * agent.get_action_name_probability(env, action, env.storm_bridge.parse_state(self.json_str))
            n_state.actions['meta_action'][next_state_id] = next_state_prob
        total_sum = 0
        for next_state_id in n_state.actions['meta_action']:
            total_sum += n_state.actions['meta_action'][next_state_id]
        #print(total_sum)
        if total_sum < 0.9:
            #print("State:",self.id)
            #print(total_sum)
            for next_state_id in n_state.actions['meta_action']:
                n_state.actions['meta_action'][next_state_id]= 1


        #exit(0)
        return n_state

class ModelChecker():
    """
    The model checker checks the product of the environment and
    the policy based on a property query.
    """

    def __init__(self, mapper: StateMapper):
        """Initialization

        Args:
            mapper (StateMapper): State Variable mapper
        """
        self.counter = 0
        self.state_id_counter = 0
        self.old_state = None
        assert isinstance(mapper, StateMapper)


    def __get_clean_state_dict(self, state_valuation_json: JsonContainerRational,
                               example_json: str) -> dict:
        """Get the clean state dictionary.

        Args:
            state_valuation_json (str): Raw state
            example_json (str): Example state as json str

        Returns:
            dict: Clean State
        """
        assert isinstance(state_valuation_json, JsonContainerRational)
        assert isinstance(example_json, str)
        state_valuation_json = json.loads(str(state_valuation_json))
        state = {}
        # print(state_valuation_json)
        # print(example_json)
        example_json = json.loads(example_json)
        for key in state_valuation_json.keys():
            for _key in example_json.keys():
                if key == _key:
                    state[key] = state_valuation_json[key]

        assert isinstance(state, dict)
        return state

    def __get_numpy_state(self, env, state_dict: dict) -> np.ndarray:
        """Get numpy state

        Args:
            env (SafeGym): SafeGym
            state_dict (dict): State as Dictionary

        Returns:
            np.ndarray: State
        """
        assert isinstance(state_dict, dict)
        state = env.storm_bridge.parse_state(json.dumps(state_dict))
        assert isinstance(state, np.ndarray)
        return state

    def __get_action_for_state(self, env, agent: common.rl_agents, state: np.array) -> str:
        """Get the action name for the current state

        Args:
            env (SafeGym): SafeGym
            agent (common.rl_agents): RL agents
            state (np.array): Numpy state

        Returns:
            str: Action name
        """
        assert str(agent.__class__).find("common.rl_agents") != -1
        assert isinstance(state, np.ndarray)
        action_index = agent.select_action(state, True)
        action_name = env.action_mapper.actions[action_index]
        assert isinstance(action_name, str)
        return action_name

    def __get_stochastic_action_for_state(self, env, agent: common.rl_agents, state: np.array) -> str:
        """Get the action name for the current state

        Args:
            env (SafeGym): SafeGym
            agent (common.rl_agents): RL agents
            state (np.array): Numpy state

        Returns:
            str: Action name
        """
        assert str(agent.__class__).find("common.rl_agents") != -1
        assert isinstance(state, np.ndarray)
        action_idizes, _ = agent.model_checking_select_action(state, self.prob_threshold)
        all_action_names = []
        for action_index in action_idizes:
            action_name = env.action_mapper.actions[action_index]
            all_action_names.append(action_name)
        return all_action_names


    def modify_model(self, model, env, agent):
        builder = stormpy.SparseMatrixBuilder(rows = 0, columns = 0, entries = 0, force_dimensions = False, has_custom_row_grouping = False)
        state_labeling = stormpy.storage.StateLabeling(model.nr_states)
        idx = 0
        max_idx = model.nr_states
        for state in model.states:
            #print("=========")
            #print(idx, "/", max_idx)
            n_state = State(state.id, json_str=model.state_valuations.get_json(state.id))
            #print("State:",state.id)
            start_time = time.time()
            for action in state.actions:
                action_name = str(list(model.choice_labeling.get_labels_of_choice(action.id))[0])
                n_state.add_action(action_name)
                for transition in action.transitions:
                    # Create State
                    action_probability = transition.value()
                    next_state = transition.column
                    n_state.add_action_probability(action_name, action_probability, next_state)

            #print("Update State", time.time() - start_time)
            n_state = n_state.get_state_update(env, agent)
            #start_time = time.time()

            visited_ids = []
            for action in state.actions:
                #action_name = str(list(model.choice_labeling.get_labels_of_choice(action.id))[0])
                # We just need to pass one time over the row, because we build an DTMC
                for transition in action.transitions:
                    # Get the transitions for next state id
                    if transition.column in visited_ids:
                        continue
                    prob = n_state.get_prob_for_next_state_of_action("meta_action", transition.column)
                    #if next_state_id == transition.column:
                    builder.add_next_value(row = int(state), column = int(transition.column), value = prob)
                    visited_ids.append(transition.column)
                #print(len(action.transitions),action.transitions)

            #print("Add row to matrix", time.time() - start_time)
            #start_time = time.time()

            for label in state.labels:
                if state_labeling.contains_label(label) == False:
                    state_labeling.add_label(label)
                state_labeling.add_label_to_state(label, state)
            #print("Add label to state", time.time() - start_time)
            #idx += 1




        transition_matrix = builder.build()
        #print(transition_matrix)

        components = stormpy.SparseModelComponents(transition_matrix=transition_matrix, state_labeling=state_labeling)
        dtmc = stormpy.storage.SparseDtmc(components)
        #print(dtmc)
        return dtmc


    def induced_markov_chain(self, agent: common.rl_agents, preprocessors, env,
                             constant_definitions: str,
                             formula_str: str, collect_label_and_states:bool=False) -> Tuple[float, int]:
        """Creates a Markov chain of an MDP induced by a policy
        and applies model checking.py

        Args:
            agent (common.rl_agents): RL policy
            env (SafeGym): SafeGym
            constant_definitions (str): Constant definitions
            formula_str (str): Property query

        Returns:
            Tuple: Tuple of the property result, model size and performance metrices
        """
        assert str(agent.__class__).find("common.rl_agents") != -1
        assert isinstance(constant_definitions, str)
        assert isinstance(formula_str, str)
        info = {}
        env.reset()
        start_time = time.time()
        prism_program = stormpy.parse_prism_program(env.storm_bridge.path)



        suggestions = dict()
        i = 0
        for module in prism_program.modules:
            for command in module.commands:
                if not command.is_labeled:
                    suggestions[command.global_index] = "tau_" + \
                        str(i)  # str(m.name)
                    i += 1

        prism_program = stormpy.preprocess_symbolic_input(
            prism_program, [], constant_definitions)[0].as_prism_program()

        prism_program = prism_program.label_unlabelled_commands(suggestions)

        properties = stormpy.parse_properties(formula_str, prism_program)
        options = stormpy.BuilderOptions([p.raw_formula for p in properties])
        #options = stormpy.BuilderOptions()
        options.set_build_state_valuations()
        options.set_build_choice_labels(True)

        collected_states = []
        collected_action_idizes = []
        collected_action_names = []

        #self.file = open("tmp_state_id_state_pair.csv","w")



        def incremental_building(state_valuation: SimpleValuation, action_index: int) -> bool:
            """Whether for the given state and action, the action should be allowed in the model.

            Args:
                state_valuation (SimpleValuation): State valuation
                action_index (int): Action index

            Returns:
                bool: Allowed?
            """
            assert isinstance(state_valuation, SimpleValuation)
            assert isinstance(action_index, int)
            simulator.restart(state_valuation)
            available_actions = sorted(simulator.available_actions())
            current_action_name = prism_program.get_action_name(action_index)
            # conditions on the action
            state = self.__get_clean_state_dict(
                state_valuation.to_json(), env.storm_bridge.state_json_example)
            state = self.__get_numpy_state(env, state)




            # Preprocess state
            if preprocessors!=None:
                for preprocessor in preprocessors:
                    state = preprocessor.preprocess(agent, state, env.action_mapper, current_action_name, True)

            # Collect states and actions if wanted
            if collect_label_and_states:
                if any(np.array_equal(state, item) for item in collected_states) == False:
                    self.counter += 1
                    collected_states.append(state)
                    collected_action_idizes.append(agent.select_action(state, True))

            # Check if actions are available
            if len(available_actions) == 0:
                return False

            cond1 = False
            # Depending on deterministic or stochastic agent, choose actions differently
            try:
                old = agent.old
            except:
                old = False
            if isinstance(agent, DeterministicAgent) or (isinstance(agent, StochasticAgent) and old):
                # If agent == deterministic
                selected_action = self.__get_action_for_state(env, agent, state)

                # Check if selected action is available
                if (selected_action in available_actions) is not True:
                    selected_action = available_actions[0]

                cond1 = (current_action_name == selected_action)
            elif isinstance(agent, StochasticAgent):
                # If agent == stochastic
                # Get list of potential actions
                all_action_names = self.__get_stochastic_action_for_state(env, agent, state)
                # For each action name that is not in available action, replace with first available action
                for i in range(len(all_action_names)):
                    if all_action_names[i] not in available_actions:
                        all_action_names[i] = available_actions[0]
                # Set cond1 true, if one of the selected actions is the current action.
                if current_action_name in all_action_names:
                    cond1 = True
            else:
                raise Exception("Agent type not supported")


            ############# This is needed to collect all the state_id-states pairs during the incremental building
            '''
            if self.old_state is None or np.array_equal(self.old_state, state) == False:
                self.old_state = state.copy()
                self.file.write(str(self.state_id_counter) + ";" + ','.join(map(str, self.old_state)) + "\n")
                self.state_id_counter+=1
            '''
            #############

            assert isinstance(cond1, bool)
            return cond1


        model_building_start = time.time()
        simulator = stormpy.simulator.create_simulator(prism_program)
        simulator.set_action_mode(
            stormpy.simulator.SimulatorActionMode.GLOBAL_NAMES)

        constructor = stormpy.make_sparse_model_builder(prism_program, options,
                                                        stormpy.StateValuationFunctionActionMaskDouble(
                                                            incremental_building))
        model = constructor.build()

        try:
            old = agent.old
        except:
            old = False
        if isinstance(agent, StochasticAgent) and old == False:
            formula_str = formula_str.replace("Pmin", "Pmax").replace("Pmax", "P")
            model = self.modify_model(model, env, agent)
            stormpy.export_to_drn(model,"test.drn")




        model_size = len(model.states)
        model_transitions = model.nr_transitions
        model_checking_start_time = time.time()


        properties = stormpy.parse_properties(formula_str, prism_program)

        result = stormpy.model_checking(model, properties[0])

        model_checking_time = time.time() - model_checking_start_time


        initial_state = model.initial_states[0]
        #print('Result for initial state', result.at(initial_state))
        mdp_result = result.at(initial_state)
        #self.file.close()
        info = {"property": formula_str, "model_building_time": (time.time()-start_time), "model_checking_time": model_checking_time, "model_size": model_size, "model_transitions": model_transitions, "collected_states": collected_states, "collected_action_idizes": collected_action_idizes}


        stormpy.export_to_drn(model,"test.drn")
        return mdp_result, info
