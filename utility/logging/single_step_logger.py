import numpy as np
from torch.utils.tensorboard import SummaryWriter

from utility.logging.base_logger import BaseLogger

class SingleStepLogger(BaseLogger):
    def __init__(self, is_logging):
        super().__init__(is_logging)
        self.logger = None

    def log_acc_reward_single_planning_step(self, test_name, step, acc_reward, actions, std=None):
        title = f'Single_Step_test/{test_name}'
        self._add_scalar(title, acc_reward, step, self.logger)
        self._add_text(title, f'mean {round(acc_reward,5)} +- {std},  actions: {actions}',
                       step=step, logger=self.logger)

    def start_log(self, name):
        if not self._is_logging:
            return
        self.logger = SummaryWriter(log_dir=f'{self.log_dir_root}/single_step/{name}')

    def commit_log(self):
        if not self._is_logging:
            return
        self.logger.flush()

    def end_log(self):
        if not self._is_logging:
            return
        self.commit_log()
        self.logger.close()
