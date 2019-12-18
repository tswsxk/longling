# coding: utf-8
# create by tongshiwei on 2019-9-4


class Module(object):
    def __init__(self, parameters):
        self.cfg = parameters

    def __str__(self):
        """
        显示模块参数
        Display the necessary params of this Module

        Returns
        -------

        """
        string = ["Params"]
        for k, v in vars(self.cfg).items():
            string.append("%s: %s" % (k, v))
        return "\n".join(string)
