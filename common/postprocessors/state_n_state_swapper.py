from common.postprocessors.postprocessor import Postprocessor
import numpy as np

class StateNStateSwapper(Postprocessor):

    def __init__(self, state_mapper, config):
        super().__init__(state_mapper, config)

    def postprocess_after_step(self, rl_agent, state, action, reward, next_state, done):
        tmp_state = state.copy()
        state = next_state.copy()
        next_state = tmp_state.copy()
        return state, action, reward, next_state, done
