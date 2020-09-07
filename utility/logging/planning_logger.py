from torch.utils.tensorboard import SummaryWriter
from utility.logging.base_logger import BaseLogger


class PlanningLogger(BaseLogger):

    def __init__(self, is_logging):
        super().__init__(is_logging)
        self._planning_test_writer = None

    def log_trial_rewards(self, test_name, trial_idx, total_reward, max_reward):
        self._add_scalar(f"{test_name}/Max reward per trial", max_reward, trial_idx)
        self._add_scalar(f"{test_name}/Total reward per trial", total_reward, trial_idx)

    def log_custom_trial_results(self, test_name, trial_idx, results):
        self._add_text(tag=f'{test_name}/results ', value=results, step=trial_idx)

    def log_agent_settings(self, test_name, agent, settings):
        self._add_text(tag=f'{test_name}/{agent}', value=settings, step=0)

    def log_iteration_max_reward(self, test_name, trials, iteration, max_reward):
        title = f"{test_name}/Average Max reward  of {trials} trials"
        self._add_scalar(title, max_reward, iteration)

    def log_iteration_avg_reward(self, test_name, trials, iteration, avg_reward):
        title = f"{test_name}/Average Total reward of {trials} trials"
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

    def _add_text(self, tag, value, step):
        if not self._is_logging:
            return
        self._planning_test_writer.add_text(tag=tag, text_string=value, global_step=step)

    def _add_scalar(self, tag, value, step):
        if not self._is_logging:
            return
        self._planning_test_writer.add_scalar(tag, value, step)


