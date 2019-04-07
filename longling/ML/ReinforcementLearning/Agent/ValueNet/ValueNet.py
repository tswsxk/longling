# coding:utf-8
# created by tongshiwei on 2018/11/8


class ValueNet(object):
    def __init__(self):
        pass

    def begin_epoch(self):
        raise NotImplementedError

    def step_epoch(self):
        raise NotImplementedError

    def end_epoch(self):
        raise NotImplementedError

    def begin_episode(self):
        raise NotImplementedError

    def step_fit(self):
        raise NotImplementedError

    def end_episode(self):
        raise NotImplementedError

    def copy_net(self, *args, **kwargs):
        raise NotImplementedError
