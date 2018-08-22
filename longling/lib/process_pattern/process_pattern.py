# coding:utf-8
# todo 文件说明写成 README.md
"""
此文件用以处理文件正则，需要指定一个正则规则文件
文件正则文件各式如下

不可命中规则1 不可命中规则2: 命中规则1 命中规则2

规则之间由空格分割，不可命中可以有也可以没有，不可命中和可命中规则均可有多个
在 分号: 前的规则有一个被命中视为与该正则规则不符， 分号: 后的规则有一个被命中，视为符合

规则实例文件见example/process_pattern_regex.exp
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from longling.base import *
from longling.lib.candylib import type_assert
from longling.lib.process_pattern.pattener import line_init_patterns, init_patterns
from longling.lib.process_pattern.process_pattern_base import ProcessPatternEncodedError


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
    >>> p = PatternHitter(":世界 和平", 1)
    >>> print(p.hit_num("世界 和平"))
    1
    >>> print(p.is_in(u"世界和平"))
    True
    >>> p = PatternHitter(":世界 和平", 0)
    >>> print(p.hit_num("世界 和平"))
    1
    >>> print(p.is_in(u"世界和平"))
    False
    """
    @type_assert(mode=int)
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

    def is_in(self, line):
        """
        检查输入中是否命中规则

        Parameters
        ----------
        line: string_type

        Returns
        -------
        bool
            命中返回True
        """
        if isinstance(line, str):
            line = unistr(line)
        if not isinstance(line, string_types):
            raise Exception("输入必须是str或unicode")
        for name, (ps1, ps2) in self.pps.items():
            t1 = regex(ps1, line)
            t2 = regex(ps2, line)
            if (len(ps1) == 0 or t1 is not True) and t2 is True:  # 不满足1的同时满足2才行
                # print line.encode('utf-8'),name.encode('utf-8')
                return True
        return False

    def hit_num(self, line):
        """
        计算输入命中规则的次数

        Parameters
        ----------
        line: string_type

        Returns
        -------
        int
            命中次数
        """
        if isinstance(line, str):
            line = unistr(line)
        if not isinstance(line, string_types):
            raise ProcessPatternEncodedError("输入必须是str或unicode")
        t = 0
        for name, (ps1, ps2) in self.pps.items():
            t1 = regex(ps1, line)
            t2 = regex(ps2, line)
            if (len(ps1) == 0 or t1 != True) and t2 == True:  # 不满足1的同时满足2才行
                t += 1
        return t

    def hit_group(self, line):
        hit_group = []
        for name, (ps1, ps2) in self.pps.items():
            t1 = regex(ps1, line)
            t2 = regex(ps2, line)
            if (len(ps1) == 0 or t1 is not True) and t2 is True:  # 不满足1的同时满足2才行
                hit_group.append(name)
        return hit_group

        # def hit_num(self, line):
        #     return len(self.hit_group(line))


if __name__ == '__main__':
    import doctest

    doctest.testmod()
