import os
from common.interpreter.decision_tree import *
from common.interpreter.feature_importance_ranking import *
from common.interpreter.feature_pruning import *
from common.interpreter.saliency_map import *
from common.interpreter.action_sensitivity import *
from common.interpreter.action_distribution import *
from common.interpreter.dead_end import *


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
        interpreter_name = interpreter_string.split(";")[0]
        interpreter = None
        if interpreter_name == "decision_tree":
            interpreter = DecisionTreeInterpreter(interpreter_string)
        elif interpreter_name == "feature_importance_ranking":
            interpreter = FeatureImportanceRankingInterpreter(interpreter_string)
        elif interpreter_name == "feature_pruning":
            interpreter = FeaturePruningInterpreter(interpreter_string)
        elif interpreter_name == "saliency_map":
            interpreter = SaliencyMapInterpreter(interpreter_string)
        elif interpreter_name == "action_sensitivity":
            interpreter = ActionSensitivityInterpreter(interpreter_string)
        elif interpreter_name == "action_distribution":
            interpreter = ActionDistributionInterpreter(interpreter_string)
        elif interpreter_name == "dead_end":
            interpreter = DeadEndInterpreter(interpreter_string)
        return interpreter
