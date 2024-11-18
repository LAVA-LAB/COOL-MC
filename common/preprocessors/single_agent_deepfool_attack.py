"""
This script makes use of code from the following GitHub repository: https://github.com/aminul-huq/DeepFool by aminul-huq in the preprocess method.
The code has been modified to fit the specific needs of this script.
Thank you aminul-huq for making this code available and open-source.
"""
from common.preprocessors.preprocessor import Preprocessor
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import os
import gc

class DeepFool(Preprocessor):

    def __init__(self, state_mapper, attack_config_str: str, task) -> None:
        super().__init__(state_mapper, attack_config_str, task)
        self.parse_attack_config(attack_config_str)

    def parse_attack_config(self, attack_config_str: str) -> None:
        # deepfool;overshoot;max_iter
        self.attack_name, self.overshoot, self.max_iter = attack_config_str.split(';')
        self.overshoot = float(self.overshoot)
        self.max_iter = int(self.max_iter)


    def preprocess(self, rl_agent, state:np.ndarray, action_mapper, current_action_name:str, deploy:bool) -> np.ndarray:
        if self.in_buffer(state):
            return state + self.buffer[str(state)]
        else:
            state = torch.from_numpy(state).float()
            num_classes = action_mapper.get_action_count()
            f_image = rl_agent.q_eval.forward(state).data.numpy().flatten()
            I = (np.array(f_image)).flatten().argsort()[::-1]

            I = I[0:num_classes]
            label = I[0]

            input_shape = state.detach().numpy().shape
            pert_image = state.clone()
            w = np.zeros(input_shape)
            r_tot = np.zeros(input_shape)

            loop_i = 0

            x = torch.tensor(pert_image[None, :],requires_grad=True)

            fs = rl_agent.q_eval.forward(x[0])
            fs_list = [fs[I[k]] for k in range(num_classes)]
            k_i = label

            while k_i == label and loop_i < self.max_iter:

                pert = np.inf
                fs[I[0]].backward(retain_graph=True)
                grad_orig = x.grad.data.numpy().copy()

                for k in range(1, num_classes):

                    #x.zero_grad()

                    fs[I[k]].backward(retain_graph=True)
                    cur_grad = x.grad.data.numpy().copy()

                    # set new w_k and new f_k
                    w_k = cur_grad - grad_orig
                    f_k = (fs[I[k]] - fs[I[0]]).data.numpy()

                    pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

                    # determine which w_k to use
                    if pert_k < pert:
                        pert = pert_k
                        w = w_k

                # compute r_i and r_tot
                # Added 1e-4 for numerical stability
                r_i =  (pert+1e-4) * w / np.linalg.norm(w)
                r_tot = np.float32(r_tot + r_i)

                pert_image = state + (1+self.overshoot)*torch.from_numpy(r_tot)

                x = torch.tensor(pert_image, requires_grad=True)
                fs = rl_agent.q_eval.forward(x[0])
                k_i = np.argmax(fs.data.numpy().flatten())

                loop_i += 1


            r_tot = (1+self.overshoot)*r_tot

            #return r_tot, loop_i, label, k_i, pert_image
            adv_perturbation = pert_image - state
            #print(rl_agent.select_action(state), rl_agent.select_action(pert_image.numpy()))
            self.update_buffer(state.long().numpy(), adv_perturbation.numpy(), True)

            return pert_image.numpy()[0]
