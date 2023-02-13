from common.postprocessors.postprocessor import Postprocessor
import numpy as np

class RandomDone(Postprocessor):

    def __init__(self, state_mapper, config):
        super().__init__(state_mapper, config)

    def postprocess(self, rl_agent, state, action, reward, next_state, done):
        done = np.random.choice([True, False])
        return state, action, reward, next_state, done
