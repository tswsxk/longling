# coding: utf-8

"""日志设定文件"""

import logging

import json
from longling.base import string_types
from longling.lib.stream import build_dir, wf_open, wf_close

__all__ = ["LogLevel", "config_logging", "LogRF", "NullLogRF", "JsonLogRF"]


class LogLevel(object):
    ERROR = logging.ERROR
    WARN = logging.WARN
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    CRITICAL = logging.CRITICAL


LogLevelDict = {}
for key, value in {
    "error": LogLevel.ERROR,
    "warn": LogLevel.WARN,
    "info": LogLevel.INFO,
    "debug": LogLevel.DEBUG,
    "critical": LogLevel.CRITICAL,
}.items():
    LogLevelDict[key] = value
    LogLevelDict[key.upper()] = value


def config_logging(filename=None,
                   log_format='%(name)s, %(levelname)s %(message)s',
                   level=logging.INFO,
                   logger=None, console_log_level=None, propagate=False,
                   mode='a',
                   file_format=None):
    """
    主日志设定文件

    Parameters
    ----------
    filename: str or None
        日志存储文件名，不为空时将创建文件存储日志
    log_format: str
        默认日志输出格式
    level: str or int
        默认日志等级
    logger: str or logging.logger
        日志logger名，可以为空（使用root logger），
        字符串类型（创建对应名logger），logger
    console_log_level: str, int or None
        屏幕日志等级，不为空时，使能屏幕日志输出
    propagate: bool
    mode: str
    file_format: str or None
        文件日志输出格式，为空时，使用log_format
    Returns
    -------

    """
    if logger is None:
        logger = logging.getLogger()
    elif isinstance(logger, string_types):
        logger = logging.getLogger(logger)

    if isinstance(level, str):
        level = LogLevelDict[level]
    if isinstance(console_log_level, str):
        console_log_level = LogLevelDict[console_log_level]

    # need a clean state, for some module may
    # have called logging functions already (i.e. logging.info)
    # in that case, a default handler would been appended,
    # causing undesired output to stderr
    for handler in logger.handlers:
        logger.removeHandler(handler)
    formatter = logging.Formatter(log_format)
    logger.setLevel(level)
    if not propagate:
        logger.propagate = False
    if filename and filename is not None:
        build_dir(filename)
        handler = logging.FileHandler(filename, mode=mode)
        file_formatter = formatter
        if file_format:
            file_formatter = logging.Formatter(file_format)
        handler.setFormatter(file_formatter)
        logger.addHandler(handler)
    if console_log_level is not None:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(log_format))
        ch.setLevel(console_log_level)
        logger.addHandler(ch)

    return logger


class LogRF(object):
    def __init__(self, *args, **kwargs):
        self._f = None

    def write(self, *arg, **kwargs):
        raise NotImplementedError

    add = write
    dumps = write
    log = write

    def open(self, *args, **kwargs):
        raise UserWarning("open function is not implemented")

    def close(self, *args, **kwargs):
        raise UserWarning("close function is not implemented")

    @staticmethod
    def load(*args, **kwargs):
        raise UserWarning("load function is not implemented")

    def dump(self, *args, **kwargs):
        raise UserWarning("dump function is not implemented")


class NullLogRF(LogRF):
    def write(self, *arg, **kwargs):
        pass


class JsonLogRF(LogRF):
    def __init__(self, *args, **kwargs):
        super(JsonLogRF, self).__init__(*args, **kwargs)
        self._f = self.open(*args, **kwargs)

    def open(self, *args, **kwargs):
        return wf_open(*args, **kwargs)

    def write(self, *args, **kwargs):
        print(json.dumps(*args, **kwargs), self._f)

    def close(self, *args, **kwargs):
        wf_close(self._f)
