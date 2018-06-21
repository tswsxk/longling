# coding: utf-8
import logging
import sys

from longling.base import string_types
from longling.lib.stream import build_dir

'''
日志设定文件
'''


def _config_logging(filename: string_types = None, log_format: string_types = '%(name)s, %(levelname)s %(message)s',
                    level=logging.INFO,
                    logger=None, console_log_level=None, propagate: bool = False, mode: string_types = 'a',
                    file_format: string_types = None):
    '''
    主日志设定文件
    :param filename: 日志存储文件名，不为空时将创建文件存储日志
    :param log_format: 默认日志输出格式
    :param level: 默认日志等级
    :param logger: 日志logger名，可以为空（使用root logger），字符串类型（创建对应名logger），logger
    :param console_log_level: 屏幕日志等级，不为空时，使能屏幕日志输出
    :param propagate:
    :param mode:
    :param file_format: 文件日志输出格式，为空时，使用log_format
    :return:
    '''
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
    config_logging = _config_logging
