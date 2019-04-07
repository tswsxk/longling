# coding:utf-8
# created by tongshiwei on 2018/11/8

from .Agent import Agent
from .Environment import Environment


class Reward(object):
    def __init__(self):
        pass

    def __call__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def begin(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def end(self):
        raise NotImplementedError


class Manager(object):
    def __init__(self, agent, environment):
        """

        Parameters
        ----------
        agent: Agent
        environment: Environment
        """
        self.agent = agent
        self.environment = environment

    def begin_episode(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def end_episode(self):
        raise NotImplementedError

    def epoch_loop(self, steps_per_epoch):
        raise NotImplementedError

    def episode_loop(self):
        raise NotImplementedError

    def train(self, epoch_num, steps_per_epoch, *args, **kwargs):
        for e in epoch_num:
            self.epoch_loop(steps_per_epoch)
