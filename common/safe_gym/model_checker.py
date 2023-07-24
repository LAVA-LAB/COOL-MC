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
            selected_action = self.__get_action_for_state(env, agent, state)
            # Check if selected action is available
            if (selected_action in available_actions) is not True:
                selected_action = available_actions[0]


            cond1 = (current_action_name == selected_action)
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
        model_size = len(model.states)
        model_transitions = model.nr_transitions
        model_checking_start_time = time.time()

        properties = stormpy.parse_properties(formula_str, prism_program)

        result = stormpy.model_checking(model, properties[0], only_initial_states=False)

        model_checking_time = time.time() - model_checking_start_time

        stormpy.export_to_drn(model,"test.drn")
        initial_state = model.initial_states[0]
        #print('Result for initial state', result.at(initial_state))
        mdp_result = result.at(initial_state)


        info = {"mdp_result": mdp_result, "property": formula_str, "model_building_time": (time.time()-start_time), "model_checking_time": model_checking_time, "model_size": model_size, "model_transitions": model_transitions, "collected_states": collected_states, "collected_action_idizes": collected_action_idizes}
        '''
        print(result.__dir__())
        print(result.get_values())
        print(model.states.__len__())
        info["state_results"] = []
        for i in range(model.states.__len__()):
            c_state = model.states[i]
            #print(f"State {i}", result.at(c_state))
            info["state_results"].append(result.at(c_state))
        '''

        return mdp_result, info
