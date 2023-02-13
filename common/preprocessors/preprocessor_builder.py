import os
from common.preprocessors.normalizer import *
from common.preprocessors.feature_remapping import *
from common.preprocessors.policy_abstraction import *
# Adversarial Attacks
from common.preprocessors.single_agent_fgsm import *
from common.preprocessors.single_agent_deepfool_attack import *
from common.preprocessors.single_agent_ffgsm import *
# Countermeasures
from common.preprocessors.rounder import *
# Robustness
from common.preprocessors.integer_l1_robustness import *

'''
HOW TO ADD MORE AGENTS?
1) Create a new AGENTNAME.py with an AGENTNAME class
2) Inherit the agent-class
3) Override the methods
4) Import this py-script into this script
5) Add additional agent hyperparameters to the argparser
6) Add to build_agent the building procedure of your agent
'''
class PreprocessorBuilder():


    @staticmethod
    def build_preprocessors(preprocessor_path, command_line_arguments, observation_space, action_space,state_mapper):
        #print('Build model with', model_root_folder_path, command_line_arguments)
        #print('Environment', observation_space.shape, action_space.n)
        try:
            state_dimension = observation_space.shape[0]
        except:
            state_dimension = 1
        preprocessors = None
        if command_line_arguments['preprocessor'] != "" and command_line_arguments['preprocessor']!="None":
            preprocessors = []
        for preprocessor_str in command_line_arguments['preprocessor'].split("#"):
            preprocessor_name = preprocessor_str.split(";")[0]
            if preprocessor_name == "normalizer":
                preprocessor = Normalizer(state_mapper, preprocessor_str, command_line_arguments['task'])
                preprocessor.load(preprocessor_path)
                preprocessors.append(preprocessor)
            elif preprocessor_name == "fgsm":
                preprocessor = FGSM(state_mapper, preprocessor_str, command_line_arguments['task'])
                preprocessor.load(preprocessor_path)
                preprocessors.append(preprocessor)
            elif preprocessor_name == "deepfool":
                preprocessor = DeepFool(state_mapper, preprocessor_str, command_line_arguments['task'])
                preprocessor.load(preprocessor_path)
                preprocessors.append(preprocessor)
            elif preprocessor_name == "ffgsm":
                preprocessor = FFGSM(state_mapper, preprocessor_str, command_line_arguments['task'])
                preprocessor.load(preprocessor_path)
                preprocessors.append(preprocessor)
            elif preprocessor_name == "feature_remapping":
                preprocessor = FeatureRemapper(state_mapper, preprocessor_str, command_line_arguments['task'])
                preprocessor.load(preprocessor_path)
                preprocessors.append(preprocessor)
            elif preprocessor_name == "policy_abstraction":
                preprocessor = PolicyAbstraction(state_mapper, preprocessor_str, command_line_arguments['task'])
                preprocessor.load(preprocessor_path)
                preprocessors.append(preprocessor)
            elif preprocessor_name == "rounder":
                preprocessor = Rounder(state_mapper, preprocessor_str, command_line_arguments['task'])
                preprocessor.load(preprocessor_path)
                preprocessors.append(preprocessor)
            elif preprocessor_name == "integer_l1_robustness":
                preprocessor = IntegerL1Robustness(state_mapper, preprocessor_str, command_line_arguments['task'])
                preprocessor.load(preprocessor_path)
                preprocessors.append(preprocessor)
        return preprocessors
