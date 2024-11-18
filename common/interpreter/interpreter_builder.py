import os
from common.interpreter.decision_tree import *
from common.interpreter.critical_state_evaluator import *
from common.interpreter.nth_action_interpreter import *
from common.interpreter.action_name_ignorer_interpreter import *
from common.interpreter.temporal_critical_state_interpreter import *
from common.interpreter.what_went_wrong_interpreter import *
from common.interpreter.what_went_wrong_interpreter_prism import *
from common.interpreter.what_went_wrong_interpreter_alternative import *
from common.interpreter.temporal_feature_rank import *
from common.interpreter.temporal_feature_2d import *
from common.interpreter.temporal_action_analysis import *
from common.interpreter.temporal_adv_attacks import *
from common.interpreter.temporal_attack_and_defense import *
from common.interpreter.co_activation_graph_analysis import *
from common.interpreter.co_graph_critical_states import *
from common.interpreter.feature_pruning import *

'''
HOW TO ADD MORE INTERPRETERS?
1) Create a new INTERPRETERNAME.py with an INTERPRETERNAME class
2) Inherit the interpreter-class
3) Override the methods
4) Import this py-script into this script
5) Add to build_interpreter the building procedure of your interpreter
'''
class InterpreterBuilder():


    @staticmethod
    def build_interpreter(interpreter_string):
        #print('Build model with', model_root_folder_path, command_line_arguments)
        #print('Environment', observation_space.shape, action_space.n)
        try:
            state_dimension = observation_space.shape[0]
        except:
            state_dimension = 1
        postprocessor = None
        interpreter_name = interpreter_string.split(";")[0]
        interpreter = None
        if interpreter_name == "decision_tree":
            interpreter = DecisionTreeInterpreter(interpreter_string)
        elif interpreter_name == "critical_state_interpreter":
            interpreter = CriticalStateEvaluator(interpreter_string)
        elif interpreter_name == "nth_action_interpreter":
            interpreter = NthActionInterpreter(interpreter_string)
        elif interpreter_name == "action_name_ignorer_interpreter":
            interpreter = ActionNameIgnorerInterpreter(interpreter_string)
        elif interpreter_name == "temporal_critical_state_interpreter":
            interpreter = TemporalCriticalStateInterpreter(interpreter_string)
        elif interpreter_name == "temporal_feature_rank":
            interpreter = TemporalFeatureRank(interpreter_string)
        elif interpreter_name == "temporal_feature_2d":
            interpreter = TemporalFeature2D(interpreter_string)
        elif interpreter_name == "what_went_wrong_interpreter":
            interpreter = WhatWentWrongInterpreter(interpreter_string)
        elif interpreter_name == "what_went_wrong_interpreter_prism":
            interpreter = WhatWentWrongInterpreterPRISM(interpreter_string)
        elif interpreter_name == "what_went_wrong_interpreter_alternative":
            interpreter = WhatWentWrongInterpreterAlternative(interpreter_string)
        elif interpreter_name == "temporal_action_analysis":
            interpreter = TemporalActionAnalysis(interpreter_string)
        elif interpreter_name == "temporal_adv_attacks":
            interpreter = TemporalAdvAttacks(interpreter_string)
        elif interpreter_name == "co_activation_graph_analysis":
            interpreter = CoActivationGraphAnalysis(interpreter_string)
        elif interpreter_name == "co_graph_critical_states":
            interpreter = CoActivationGraphAnalysisCriticalStates(interpreter_string)
        elif interpreter_name == "feature_pruner":
            interpreter = FeaturePruner(interpreter_string)
        elif interpreter_name == "temporal_adv_attack_and_defense":
            interpreter = TemporalAdvAttackAndDefense(interpreter_string)
        return interpreter
