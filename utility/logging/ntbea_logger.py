from torch.utils.tensorboard import SummaryWriter
from utility.logging.base_logger import BaseLogger


class NTBEALogger(BaseLogger):
    def __init__(self, is_logging):
        super().__init__(is_logging)
        self._ntbea_writer = None

    def log_ntbea_results(self, agent, result):
        self._add_text(f'NTBEA_{agent}/Best_parameters', result, 0)

    def start_log(self, name):
        if not self._is_logging:
            return
        self._ntbea_writer = SummaryWriter(log_dir=f'{self.log_dir_root}/ntbea_results/{name}')

    def commit_log(self):
        if not self._is_logging:
            return
        self._ntbea_writer.flush()

    def end_log(self):
        if not self._is_logging:
            return
        self.commit_log()
        self._ntbea_writer.close()

    def _add_text(self, tag, value, step):
        if not self._is_logging:
            return
        self._ntbea_writer.add_text(tag=tag, text_string=value, global_step=step)
