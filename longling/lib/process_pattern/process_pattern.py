# coding:utf-8
# todo 文件说明写成 README.md
"""
此文件用以处理文件正则，需要指定一个正则规则文件
文件正则文件格式如下

0 号模式
不可命中规则: 命中规则

1 号模式
不可命中规则1 不可命中规则2: 命中规则1 命中规则2

规则之间由空格分割，不可命中可以有也可以没有，不可命中和可命中规则均可有多个
在 冒号: 前的规则有一个被命中视为与该正则规则不符，
冒号: 后的规则有一个被命中，视为符合
"""

from longling.lib.process_pattern.pattener import line_init_patterns, init_patterns

__all__ = ["PatternHitter"]


def regex(ps, line):
    if not ps:
        return False
    xs = []
    for p in ps:
        x = p.search(line)
        xs.append(x)
    for x in xs:
        if x is None:
            return False
    return True


class PatternHitter(object):
    """
    模式识别器

    Examples
    --------
    >>> p = PatternHitter("黑色毛衣:世界 和平", 1)
    >>> print(p.hit_num("世界 和平"))
    1
    >>> print(p.is_in("世界和平"))
    True
    >>> p = PatternHitter("止战之殇:世界 和平", 0)
    >>> print(p.hit_num("世界 和平"))
    1
    >>> print(p.is_in("世界和平"))
    False
    """

    def __init__(self, location, mode=0):
        """

        Parameters
        ----------
        location
        mode
        """
        if ':' in location:
            self.pps = dict()
            line_init_patterns(location, mode, self.pps)
        else:
            self.pps = init_patterns(location, mode)

    def is_in(self, line: str):
        """
        检查输入中是否命中规则

        Parameters
        ----------
        line: str

        Returns
        -------
        bool
            命中返回True
        """

        for name, (ps1, ps2) in self.pps.items():
            t1 = regex(ps1, line)
            t2 = regex(ps2, line)
            if (len(ps1) == 0 or t1 is not True) and t2 is True:
                # 不满足1的同时满足2才行
                # print line.encode('utf-8'),name.encode('utf-8')
                return True
        return False

    def hit_num(self, line: str):
        """
        计算输入命中规则的次数

        Parameters
        ----------
        line: str

        Returns
        -------
        int
            命中次数
        """

        t = 0
        for name, (ps1, ps2) in self.pps.items():
            t1 = regex(ps1, line)
            t2 = regex(ps2, line)
            if (len(ps1) == 0 or t1 is not True) and t2 is True:
                # 不满足1的同时满足2才行
                t += 1
        return t

    def hit_group(self, line):
        hit_group = []
        for name, (ps1, ps2) in self.pps.items():
            t1 = regex(ps1, line)
            t2 = regex(ps2, line)
            if (len(ps1) == 0 or t1 is not True) and t2 is True:
                # 不满足1的同时满足2才行
                hit_group.append(name)
        return hit_group
