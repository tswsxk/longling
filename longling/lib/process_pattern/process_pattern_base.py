from longling.lib.utilog import config_logging, LogLevel

logger = config_logging(
    logger='process_pattern', console_log_level=LogLevel.WARN
)


class ProcessPatternNotExistedPatternError(Exception):
    pass


class ProcessPatternLineError(Exception):
    pass


class ProcessPatternEncodedError(Exception):
    pass
