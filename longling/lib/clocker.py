# coding: utf-8

import time


class Clocker(object):
    def __init__(self):
        self.process_st = 0
        self.process_et = 0
        self.wall_st = 0
        self.wall_et = 0
        self.unit = 's'

    def start(self):
        self.process_st = time.clock()
        self.wall_st = time.time()
        return self.process_st

    def end(self, wall=False):
        self.process_et = time.clock()
        self.wall_et = time.time()
        if wall:
            return self.wall_time
        else:
            return self.process_time

    @property
    def wall_time(self):
        return self.wall_et - self.wall_st

    @property
    def process_time(self):
        return self.process_et - self.process_st


if __name__ == '__main__':
    clocker = Clocker()
    clocker.start()
    time.sleep(3)
    print(clocker.end())
