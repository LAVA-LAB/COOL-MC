import os
from common.behavioral_cloning_dataset.raw_dataset import RawDataset

'''
HOW TO ADD MORE INTERPRETERS?
1) Create a new BEHAVIORAL_CLONING_DATASET.py with an BEHAVIORAL_CLONING_DATASET_NAME class
2) Inherit the BEHAVIORAL_CLONING_DATASET-class
3) Override the methods
4) Import this py-script into this script
5) Add to build_interpreter the building procedure of your interpreter
'''
class BehavioralCloningDatasetBuilder():


    @staticmethod
    def build(config):
        behavioral_cloning_dataset_name = config.split(";")[0]
        dataset = None
        if behavioral_cloning_dataset_name == "raw_dataset":
            dataset = RawDataset(config)
        return dataset
