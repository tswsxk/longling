# coding: utf-8
# created by tongshiwei on 18-1-24

from __future__ import print_function

import logging
from longling.lib.clock import Clock


# at the beginning of each batch, the tic method will be called
# at the end of each batch, the toc_print method will be called

class BaseMonitor(object):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def tic(self):
        raise NotImplementedError

    def toc_print(self):
        raise NotImplementedError


class TimeMonitor(Clock, BaseMonitor):
    def __init__(self, logger=logging, frequent=50, print_total=True):
        super(TimeMonitor, self).__init__()
        self.logger = logger
        self.total_wall_time = 0
        self.total_process_time = 0
        self.frequent = frequent
        self.cnt_step = 0
        self.print_total = print_total

    def __call__(self, *args, **kwargs):
        self.end()
        wall_time, process_time = self.wall_time, self.process_time
        self.total_wall_time += wall_time
        self.total_process_time += process_time
        self.start()

    def tic(self):
        self.start()

    def toc_print(self):
        self.end()
        wall_time, process_time = self.wall_time, self.process_time
        self.total_wall_time += wall_time
        self.total_process_time += process_time
        self.cnt_step += 1
        if self.cnt_step % self.frequent == 0:
            self.cnt_step = 0
            msg = "{}: {}s, {}: {}s"
            if self.print_total:
                msg = msg.format('total_wall_time', self.total_wall_time, 'total_process_time', self.total_process_time)
            else:
                msg = msg.format('wall_time', wall_time, 'process_time', process_time)
            self.logger.debug(msg)

    def install(self, *args, **kwargs):
        pass
