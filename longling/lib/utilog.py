# coding: utf-8

"""日志设定文件"""

import logging
import sys

from longling.base import string_types
from longling.lib.candylib import type_assert
from longling.lib.stream import build_dir


class LogLevel(object):
    ERROR = logging.ERROR
    WARN = logging.WARN
    INFO = logging.INFO
    DEBUG = logging.DEBUG
    CRITICAL = logging.CRITICAL


def _config_logging(filename=None, log_format='%(name)s, %(levelname)s %(message)s',
                    level=logging.INFO,
                    logger=None, console_log_level=None, propagate=False, mode='a',
                    file_format=None):
    """
    主日志设定文件

    Parameters
    ----------
    filename: str or None
        日志存储文件名，不为空时将创建文件存储日志
    log_format: str
        默认日志输出格式
    level: int
        默认日志等级
    logger: str or logging.logger
        日志logger名，可以为空（使用root logger），字符串类型（创建对应名logger），logger
    console_log_level: int or None
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
    # need a clean state, for some module may have called logging functions already (i.e. logging.info)
    # in that case, a default handler would been appended, causing undesired output to stderr
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


# type checker
if sys.version_info[0] == 3:
    def config_logging(filename: string_types = None, log_format: string_types = '%(name)s, %(levelname)s %(message)s',
                       level=logging.INFO,
                       logger=None, console_log_level=None, propagate: bool = False, mode: string_types = 'a',
                       file_format: string_types = None):
        return _config_logging(filename, log_format, level, logger, console_log_level, propagate, mode, file_format)
else:
    config_logging = type_assert(filename=string_types,
                                 log_format=string_types,
                                 propagate=bool,
                                 mode=string_types,
                                 file_format=string_types)(_config_logging)
