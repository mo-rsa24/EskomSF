# Base interface
class BaseLogger:
    def log(self, record: dict):
        raise NotImplementedError("logger must implement `log()`")

    def flush(self):
        pass  # Only needed for Spark