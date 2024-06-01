# coding: utf-8
# tongshiwei@2024/6/1
import datetime
import logging
import math
import os
from contextlib import contextmanager

import psutil
from loguru import logger


@contextmanager
def timeit(prefix, level="info", hint_at_beginning=False, logger=logging, disable=False):
    if disable:
        return
    if hint_at_beginning:
        eval(f"logger.{level}")(f"Start [TIMEIT] {prefix}")
    now = datetime.datetime.now()
    yield
    eval(f"logger.{level}")(f"[TIMEIT] {prefix}, Time Cost: {(datetime.datetime.now() - now).total_seconds()}s")


def get_idle_cpu_count():
    """
    Extract information about idle CPU count.

    :return:

    Example
    ------

        .. code-block:: python
        >>> get_idle_cpu_count()
        10
    """
    total_cpu_count = os.cpu_count()
    cpu_usage = psutil.cpu_percent(interval=0.001)
    idle_cpu_count = math.floor(total_cpu_count * ((100 - cpu_usage) / 100))
    return idle_cpu_count


def get_n_jobs(n_jobs):
    if n_jobs >= 1:
        return n_jobs
    elif 0 <= n_jobs < 1:
        n_jobs = math.ceil(n_jobs * get_idle_cpu_count())
    else:
        n_jobs = get_idle_cpu_count()
    return max(n_jobs, 1)


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # 获取对应的 Loguru 等级
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 将日志记录转发给 loguru
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def config_logging(name=None, level="INFO"):
    logging.getLogger(name).addHandler(InterceptHandler())
    logging.getLogger(name).setLevel(level.upper())
