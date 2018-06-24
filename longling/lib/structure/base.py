# coding: utf-8
# create by tongshiwei on 2018/6/24
from longling.lib.utilog import config_logging, LogLevel

logger = config_logging(logger="graph", console_log_level=LogLevel.INFO)


class AddList(list):
    def add(self, elem):
        return self.append(elem)
