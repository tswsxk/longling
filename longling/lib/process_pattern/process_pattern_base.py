import logging
from longling.lib.utilog import config_logging

logger = config_logging(logger='process_pattern', console_log_level=logging.WARN)


class ProcessPatternNotExistedPatternError(Exception):
    pass


class ProcessPatternLineError(Exception):
    pass


class ProcessPatternEncodedError(Exception):
    pass
