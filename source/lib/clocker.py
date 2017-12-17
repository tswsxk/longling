# coding: utf-8

import time


class Clocker(object):
    def __init__(self):
        self.st = 0
        self.et = 0
        self.unit = 's'

    def start(self):
        self.st = time.clock()
        return self.st

    def end(self):
        self.et = time.clock()
        return self.et - self.st

    def interval(self):
        return self.et - self.st


if __name__ == '__main__':
    clocker = Clocker()
    clocker.start()
    time.sleep(3)
    print(clocker.end())
