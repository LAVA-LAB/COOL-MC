from common.preprocessors.preprocessor import Preprocessor
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import os

class FGSM(Preprocessor):

    def __init__(self, state_mapper, attack_config_str: str, task) -> None:
        super().__init__(state_mapper, attack_config_str, task)
        self.parse_attack_config(attack_config_str)

    def parse_attack_config(self, attack_config_str: str) -> None:
        # fgsm;1
        attack_name, epsilon = attack_config_str.split(';')
        self.attack_name = attack_name
        self.epsilon = float(epsilon)


    def preprocess(self, rl_agent, state:np.ndarray, action_mapper, current_action_name:str, deploy:bool) -> np.ndarray:
        if self.in_buffer(state):
            return state + self.buffer[str(state)]
        else:
            # Numpy array to torch array with requires_grad=True
            state = torch.from_numpy(state).float().requires_grad_(True)
            output = rl_agent.q_eval(state).reshape(1, -1)
            target = output.argmax().reshape(-1)
            loss = F.nll_loss(output, target)
            rl_agent.q_eval.zero_grad()
            loss.backward()
            data_grad = state.grad.data
            adv_perturbation = data_grad.sign()
            adv_perturbation = adv_perturbation.numpy()
            # clip adv_perturbation to epsilon
            state = state.detach().numpy()
            adversarial_state = state + adv_perturbation * self.epsilon
            self.update_buffer(state, adv_perturbation, True)
            return adversarial_state
