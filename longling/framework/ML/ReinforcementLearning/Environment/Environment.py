# coding:utf-8
# created by tongshiwei on 2018/11/8


class Environment(object):
    def __init__(self):
        pass

    def begin_episode(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def reward(self):
        raise NotImplementedError

    def end_episode(self):
        raise NotImplementedError
