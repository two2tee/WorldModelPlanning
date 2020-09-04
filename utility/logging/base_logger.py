

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
