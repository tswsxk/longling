# coding: utf-8
# create by tongshiwei on 2019/6/27

from longling.lib.stream import AddObject


class NullDevice(AddObject):
    def add(self, value):
        pass
