""" RMHC for simulated environments """
#  Copyright (c) 2020, - All Rights Reserved
#  This file is part of the Evolutionary Planning on a Learned World Model thesis.
#  Unauthorized copying of this file, via any medium is strictly prohibited without the consensus of the authors.
#  Written by Thor V.A.N. Olesen <thorolesen@gmail.com> & Dennis T.T. Nguyen <dennisnguyen3000@yahoo.dk>.

import torch
from planning.interfaces.abstract_grad_hill_climb_simulation import AbstractGradientHillClimbing
from utility.logging.single_step_logger import SingleStepLogger


class SGDHC(AbstractGradientHillClimbing):
    def __init__(self, horizon, max_steps, is_shift_buffer, learning_rate):
        super().__init__(horizon, max_steps, is_shift_buffer, learning_rate)
        self.logit_sequence = None
        self.latent = None
        self.hidden = None
        self.optimizer = None

    def search(self, environment, latent, hidden):
        self.latent = latent
        self.hidden = hidden
        self.logit_sequence = self._initialize_sequence(environment)

        self.optimizer = torch.optim.Adam(self.logit_sequence, lr=self.learning_rate)

        # logger = SingleStepLogger(is_logging=True) # TODO REMOVE
        # logger.start_log(f'World_Model_RandomNormal_SGDHC_h{self.horizon}_g{self.max_steps}_lr{self.learning_rate}')

        for step in range(self.max_steps):
            plan = [environment.convert_logits_to_action(logit) for logit in self.logit_sequence]
            total_reward = self._evaluate_plan(plan, environment)
            self._gradient_step(total_reward)
            # print(total_reward, self.logit_sequence)
            # logger.log_acc_reward_single_planning_step(test_name='planning_head_to_grass_right', step=step, acc_reward=total_reward.item(), actions=[self._convert_action_to_value(action) for action in plan])
        # logger.end_log()

        best_action = self._convert_action_to_value(environment.convert_logits_to_action(self.logit_sequence[0]))
        return best_action, None


    def _initialize_sequence(self, environment):
        if self.is_shift_buffer and self.logit_sequence is not None:
            sequence = self._shift_buffer(environment, self.logit_sequence)
        else:
            sequence = [torch.tensor(environment.sample_logits(), requires_grad=True) for _ in range(self.horizon)]
        return sequence

    def _convert_action_to_value(self, action):
        return [sub_action.item() for sub_action in action]

    def _convert_tensor_sequence_to_sequence(self, sequence):
        return [self._convert_action_to_value(action) for action in sequence]

    def _shift_buffer(self, environment, sequence):
        sequence.pop(0)
        sequence.append(torch.tensor(environment.sample_logits(), requires_grad=True))
        return sequence

    def _gradient_step(self,  total_reward):
        self.optimizer.zero_grad()
        (-total_reward).backward(retain_graph=True)  # TODO check mem leak
        self.optimizer.step()

    def _evaluate_plan(self, actions, environment):
        is_done = False
        total_reward = 0
        latent = self.latent
        hidden = self.hidden

        for action in actions:
            if not is_done:
                latent, reward, is_done, hidden = environment.step(action, hidden, latent, is_simulation_real_environment=False, is_reward_tensor=True)
                total_reward += reward
            else:
                break
        return total_reward
