# coding:utf-8
# created by tongshiwei on 2018/11/8
import random

class Agent(object):
    def __init__(self,  *args, **kwargs):
        pass

    def begin_episode(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def end_episode(self):
        raise NotImplementedError

    def learning(self):
        raise NotImplementedError

    def exploitation(self):
        raise NotImplementedError

    def exploration(self):
        raise NotImplementedError

    def pi(self, epsilon=0):
        if random.random() < 1 - epsilon:
            self.exploitation()
        else:
            self.exploration()
        raise NotImplementedError

    def state_transform(self):
        raise NotImplementedError

    def get_candidates_actions(self):
        pass

class ValueAgent(Agent):
    pass


class PolicyAgent(Agent):
    pass
