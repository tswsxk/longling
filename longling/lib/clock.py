# coding: utf-8
from __future__ import absolute_import

import time

import logging
from longling.lib.utilog import config_logging

logger = config_logging(logger="clock", console_log_level=logging.INFO)


class Clock(object):
    '''
    计时器
    单位：秒
    '''

    def __init__(self, store_dict=None, logger=logger, tips=''):
        assert store_dict is None or type(store_dict) is dict
        self.process_st = 0
        self.process_et = 0
        self.wall_st = 0
        self.wall_et = 0
        self.store_dict = store_dict
        self.logger = logger
        self.tips = tips

    def start(self):
        '''
        开始计时
        :return:
        '''
        self.process_st = time.clock()
        self.wall_st = time.time()
        return self.process_st

    def end(self, wall=False):
        '''
        计时结束，返回间隔时间
        :param wall:
        :return:
        '''
        self.process_et = time.clock()
        self.wall_et = time.time()
        if wall:
            return self.wall_time
        else:
            return self.process_time

    @property
    def wall_time(self):
        '''
        获取程序运行时间（包括等待时间）
        :return:
        '''
        return self.wall_et - self.wall_st

    @property
    def process_time(self):
        '''
        获取程序运行时间（不包括等待时间）
        :return:
        '''
        return self.process_et - self.process_st

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.info(self.end() if not self.tips else '%s %s' % (self.tips, self.end()))
        if self.store_dict is not None:
            self.store_dict['wall_time'] = self.wall_time
            self.store_dict['process_time'] = self.process_time


if __name__ == '__main__':
    with Clock():
        time.sleep(1)
