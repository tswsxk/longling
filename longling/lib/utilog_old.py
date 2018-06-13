# coding: utf-8
import logging
import time
import os

MIDNIGHT = 24 * 60 * 60  # 00:00:00
SECONDS_PER_DAY = 60 * 60 * 24


from longling.base import string_types
from longling.lib.stream import build_dir
'''
日志设定文件
'''


# def keyboard_change_level(logger):
#     import signal
#     def warn_logging_level(signum, frame):
#         logger.setLevel(logging.INFO)
#
#     def info_logging_level(signum, frame):
#         logger.setLevel(logging.WARN)
#
#     signal.signal(signal.SIGUSR1, warn_logging_level)
#     signal.signal(signal.SIGUSR2, info_logging_level)


# def logger_params_set(logger, params):
#     if "level" in params:
#         logger.setLevel(params['level'])
#
#
# def file_and_screen_logger(filename, file_params={}, console_params={}, logger_name=""):
#     logger = logging.getLogger(logger_name)
#     console = logging.StreamHandler()
#     logger_params_set(console, console_params)
#     filer = logging.FileHandler(filename)
#     logger_params_set(filer, file_params)
#     logger.addHandler(console)
#     logger.addHandler(filer)
#     return logger
#
#
# def simple_stream_logger(logger_name, logger_level=logging.WARN, format='%(asctime)s, %(levelname)s %(message)s'):
#     logger = logging.getLogger(logger_name)
#     sh = logging.StreamHandler()
#     if format:
#         sh.setFormatter(logging.Formatter(format))
#     logger.addHandler(sh)
#     logger.setLevel(logger_level)
#     logger.propagate = False
#     return logger


# '%(asctime)s, %(levelname)s %(message)s'

def config_logging(filename=None, format='%(name)s, %(levelname)s %(message)s', level=logging.INFO,
                   logger=None, console_log_level=None, propagate=False, mode='a',
                   file_format=None):
    if logger is None:
        logger = logging.getLogger()
    elif isinstance(logger, string_types):
        logger = logging.getLogger(logger)
    # need a clean state, for some module may have called logging functions already (i.e. logging.info)
    # in that case, a default handler would been appended, causing undesired output to stderr
    for handler in logger.handlers:
        logger.removeHandler(handler)
    formatter = logging.Formatter(format)
    logger.setLevel(level)
    if not propagate:
        logger.propagate = False
    if filename and filename is not None:
        # if 'when' not in multi_process_logger_kwargs:
        #     multi_process_logger_kwargs['when'] = 'midnight'
        # handler = MultiProcessRotatingFileHandler(filename=filename, **multi_process_logger_kwargs)
        build_dir(filename)
        handler = logging.FileHandler(filename, mode=mode)
        file_formatter = formatter
        if file_format:
            file_formatter = logging.Formatter(file_format)
        handler.setFormatter(file_formatter)
        logger.addHandler(handler)
    if console_log_level is not None:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(format))
        ch.setLevel(console_log_level)
        logger.addHandler(ch)

    return logger


# from logging.handlers import TimedRotatingFileHandler
#
#
# class MultiProcessRotatingFileHandler(TimedRotatingFileHandler):
#     def __init__(self, filename, when='h', interval=1, backupCount=0, encoding=None, utc=False):
#         super(MultiProcessRotatingFileHandler, self).__init__(filename, when, interval, backupCount, encoding, False,
#                                                               utc)
#         d, f = os.path.split(filename)
#         self.lock_file_name = os.path.join(d, '.' + f)
#
#     def computeRollover(self, currentTime):
#         """
#         Work out the rollover time based on the specified time.
#         都在整数时间点rollover
#         """
#         if self.when == 'MIDNIGHT' or self.when.startswith('W'):
#             if self.utc:
#                 t = time.gmtime(currentTime)
#             else:
#                 t = time.localtime(currentTime)
#             currentHour = t[3]
#             currentMinute = t[4]
#             currentSecond = t[5]
#             secondsToMidnight = MIDNIGHT - ((currentHour * 60 + currentMinute) * 60 +
#                                             currentSecond)
#             result = currentTime + secondsToMidnight
#             if self.when.startswith('W'):
#                 day = t[6]
#                 if day != self.dayOfWeek:
#                     if day < self.dayOfWeek:
#                         daysToWait = self.dayOfWeek - day
#                     else:
#                         daysToWait = self.dayOfWeek - day + 7
#                     result += SECONDS_PER_DAY * daysToWait
#         else:
#             result = currentTime + self.interval - currentTime % self.interval
#         return result
#
#     def doRollover(self):
#         from filelock import FileLock
#         """
#         do a rollover; in this case, a date/time stamp is appended to the filename
#         when the rollover happens.  However, you want the file to be named for the
#         start of the interval, not the current time.  If there is a backup count,
#         then we have to get a list of matching filenames, sort them and remove
#         the one with the oldest suffix.
#         rollover时两种情况:
#         1、检查要被rename to的文件是否已经存在，如果已经存在，说明有另外的进程已经rollover了，那本进程
#             a）reopen
#             b) 更新rollover时间
#         2、如果不存在，说明本进程是最先抢到rollover锁的，那本进程：
#             a) rename
#             b) 删除旧日志
#             c) reopen
#             d) 更新rollover时间
#         """
#         if self.stream:
#             self.stream.close()
#             self.stream = None
#         # get the time that this sequence started at and make it a TimeTuple
#         currentTime = int(time.time())
#         t = self.rolloverAt - self.interval
#         if self.utc:
#             timeTuple = time.gmtime(t)
#         else:
#             timeTuple = time.localtime(t)
#         dfn = self.baseFilename + "." + time.strftime(self.suffix, timeTuple)
#
#         with FileLock(self.lock_file_name):
#             if not os.path.exists(dfn):
#                 self._rollover(dfn)
#         self.stream = self._open()
#         self._updateRolloverAt(currentTime)
#
#     def _rollover(self, dfn):
#         # Issue 18940: A file may not have been created if delay is True.
#         if os.path.exists(self.baseFilename):
#             os.rename(self.baseFilename, dfn)
#
#         if self.backupCount > 0:
#             for s in self.getFilesToDelete():
#                 os.remove(s)
#
#     def _updateRolloverAt(self, currentTime):
#         """
#         更新下一次rollover的时间点
#         :param currentTime:
#         :return:
#         """
#         newRolloverAt = self.computeRollover(currentTime)
#         while newRolloverAt <= currentTime:
#             newRolloverAt = newRolloverAt + self.interval
#         self.rolloverAt = newRolloverAt