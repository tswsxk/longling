# coding: utf-8
# create by tongshiwei on 2019-9-4


__all__ = ["Module"]


class Module(object):
    """
    Examples
    --------
    >>> class Params:
    ...     def __init__(self):
    ...         self.a = 1
    ...         self.b = 2
    >>> params = Params()
    >>> print(Module(params))
    Params
    a: 1
    b: 2
    """

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
