

class BaseLogger:
    def __init__(self, is_logging):
        self.log_dir_root = 'utility/logging/tensorboard_runs'
        self._is_logging = is_logging


    def start_log(self, name):
        pass

    def commit_log(self):
        pass

    def end_log(self):
        pass

    def _add_text(self, tag, value, step, logger):
        if not self._is_logging:
            return
        logger.add_text(tag=tag, text_string=value, global_step=step)

    def _add_scalar(self, tag, value, step, logger):
        if not self._is_logging:
            return
        logger.add_scalar(tag, value, step)
