from torch.utils.tensorboard import SummaryWriter

from utility.logging.base_logger import BaseLogger


class PlanningLogger(BaseLogger):

    def __init__(self, is_logging):
        super().__init__(is_logging)
        self._planning_test_writer = None


    def log_iteration_max_reward(self, name, iteration, max_reward):
        title = f"{name}/Average Max reward"
        self._add_scalar(title, max_reward, iteration)

    def log_iteration_avg_reward(self, name, iteration, avg_reward):
        title = f"{name}/Average Total reward"
        self._add_scalar(title, avg_reward, iteration)

    def start_log(self, name):
        if not self._is_logging:
            return
        self._planning_test_writer = SummaryWriter(log_dir=f'{self.log_dir_root}/planning_test/{name}')

    def commit_log(self):
        if not self._is_logging:
            return
        self._planning_test_writer.flush()

    def end_log(self):
        if not self._is_logging:
            return
        self.commit_log()
        self._planning_test_writer.close()

    def _add_scalar(self, tag, value, step):
        if not self._is_logging:
            return
        self._planning_test_writer.add_scalar(tag, value, step)


