# coding: utf-8
# create by tongshiwei on 2019/4/8

__all__ = ["AttrDict"]


class AttrDict(dict):
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, item):
        return self[item]

    def __delattr__(self, item):
        del self[item]
