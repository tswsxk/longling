# coding:utf-8
# created by tongshiwei on 2018/11/8


class Agent(object):
    def __init__(self, *args, **kwargs):
        pass

    def begin_episode(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def end_episode(self):
        raise NotImplementedError


class ValueAgent(Agent):
    pass


class PolicyAgent(Agent):
    pass
