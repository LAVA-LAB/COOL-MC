import os
from common.interpreter.decision_tree import *

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
        return interpreter
