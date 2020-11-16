# coding: utf-8
from __future__ import absolute_import

import logging
import time
from contextlib import contextmanager

from longling.lib.utilog import config_logging

_logger = config_logging(logger="clock", console_log_level=logging.INFO)

__all__ = ["Clock", "print_time", "Timer"]


@contextmanager
def print_time(tips: str = "", logger=_logger):
    """
    统计并打印脚本运行时间， 秒为单位

    Parameters
    ----------
    tips: str
    logger: logging.Logger or logging

    Examples
    --------
    >>> with print_time("tips"):
    ...     a = 1 + 1  # The code you want to test

    """
    start_time = time.time()
    logger.info('Starting to %s', tips)
    yield
    logger.info('Finished to {} in {:.6f} seconds'.format(
        tips,
        time.time() - start_time)
    )


class Clock(object):
    r"""
    计时器。
    包含两种时间：wall_time 和 process_time

    * wall_time: 包括等待时间在内的程序运行时间
    * process_time: 不包括等待时间在内的程序运行时间

    Parameters
    ----------
    store_dict: dict or None
        with closure 中存储运行时间
    logger: logging.logger
        日志
    tips: str
        提示前缀

    Examples
    --------
    .. code-block :: python

        with Clock():
            a = 1 + 1
        clock = Clock()
        clock.start()
        # some code
        clock.end(wall=True) # default to return the wall_time, to get process_time, set wall=False

    """

    def __init__(self, store_dict: (dict, None) = None, logger: (logging.Logger, None) = _logger, tips=''):
        assert store_dict is None or type(store_dict) is dict
        self.process_st = 0
        self.process_et = 0
        self.wall_st = 0
        self.wall_et = 0
        self.store_dict = store_dict
        self.logger = logger
        self.tips = tips

    def start(self):
        """开始计时"""
        self.process_st = time.process_time()
        self.wall_st = time.time()
        return self.process_st

    def time(self):
        return time.time() - self.wall_st

    def end(self, wall=True):
        """计时结束，返回间隔时间"""
        self.process_et = time.process_time()
        self.wall_et = time.time()
        if wall:
            return self.wall_time
        else:
            return self.process_time

    @property
    def wall_time(self):
        """获取程序运行时间（包括等待时间）"""
        return self.wall_et - self.wall_st

    @property
    def process_time(self):
        """获取程序运行时间（不包括等待时间）"""
        return self.process_et - self.process_st

    def __enter__(self):
        if self.tips:
            if self.logger is not None:
                self.logger.info(self.tips)
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger is not None:
            self.logger.info(
                '%ss' % self.end() if not self.tips else '%s %ss' % (
                    self.tips, self.end()
                )
            )
        if self.store_dict is not None:
            self.store_dict['wall_time'] = self.wall_time
            self.store_dict['process_time'] = self.process_time


Timer = Clock
