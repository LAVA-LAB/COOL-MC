class Postprocessor:

    def __init__(self, state_mapper, config):
        self.state_mapper = state_mapper
        self.config = config

    def postprocess(self, rl_agent, state, action, reward, next_state, done):
        raise NotImplementedError()

    def save(self):
        """
        Saves the manipulator in the MLFlow experiment.
        """
        pass

    def load(self, root_folder:str):
        """
        Loads the manipulator from the folder

        Args:
            root_folder ([str]): Path to the folder
        """
        pass

