from common.behavioral_cloning_dataset.dataset import *


class RawDataset(BehavioralCloningDataset):

    def get_data(self):
        return {"X_train": self.X, "y_train": self.y, "X_test": None, "y_test": None}