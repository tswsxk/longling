# coding:utf-8
'''
此文件用以处理文件正则，需要指定一个正则规则文件

文件正则文件各式如下

不可命中规则1 不可命中规则2: 命中规则1 命中规则2

规则之间由空格分割，不可命中可以有也可以没有，不可命中和可命中规则均可有多个
在 分号: 前的规则有一个被命中视为与该正则规则不符， 分号: 后的规则有一个被命中，视为符合

规则实例文件见example/process_pattern_regex.exp
'''
from __future__ import print_function
from __future__ import unicode_literals

import re

from longling.base import *


class cond_filter():
    # 过滤器，过滤 只含表情，数字 以及 回复（含有 //@ ）评论
    def __init__(self):
        self.emoji_re = re.compile('['
                                   '\U0001F300-\U0001F64F'
                                   '\U0001F680-\U0001F6FF'
                                   '\u2600-\u2B55]+',
                                   re.UNICODE)

    def is_valid(self, data):
        if re.sub("^\d+$", "", self.emoji_re.sub("", data['text'])):
            if "//@" not in data['text']:
                return True
        return False


def _init_patterns(line, pps=None):
    try:
        line = line.decode('utf-8').strip()
    except:
        pass
    try:
        line1, line2 = line.split(':')
    except:
        print(line)
        raise Exception('error in %s' % line)

    v = line1.strip()
    ps1 = []
    if v:
        try:
            p = re.compile(v)
        except:
            print("error:", v)
            raise Exception('error in %s' % line)
        ps1.append(p)

    v = line2.strip()
    assert v
    ps2 = []
    try:
        p = re.compile(v)
    except:
        print("error:", v)
        raise Exception('error in %s' % line)
    ps2.append(p)

    if pps is not None and isinstance(pps, dict):
        pps[line] = (ps1, ps2)
    return line, (ps1, ps2)


def init_patterns(location):
    pps = {}
    with open(location) as f:
        for line in f:
            try:
                _init_patterns(line, pps)
            except:
                continue
    return pps


def _init_patterns2(line, pps):
    try:
        line = line.decode('utf-8').strip()
    except:
        pass
    try:
        line1, line2 = line.split(':')
    except:
        print(line)
        raise Exception('error in %s' % line)

    vs = line1.split()
    ps1 = []
    for v in vs:
        try:
            p = re.compile(v)
        except:
            print("error:", v)
            raise Exception('error in %s' % line)
        ps1.append(p)

    vs = line2.split()
    ps2 = []
    for v in vs:
        try:
            p = re.compile(v)
        except:
            print("error:", v)
            raise Exception('error in %s' % line)
        ps2.append(p)

    if pps is not None and isinstance(pps, dict):
        pps[line] = (ps1, ps2)

    return line, (ps1, ps2)


def init_patterns2(location):
    '''
    创建一个模式字典
    :param location: str, 正则规则文件位置
    :return: dict, 模式字典
    '''
    pps = {}
    with open(location) as f:
        for line in f:
            try:
                _init_patterns2(line, pps)
            except:
                continue
    return pps


def regex(ps, line):
    if not ps:
        return False
    xs = []
    for p in ps:
        x = p.search(line)
        xs.append(x)
    for x in xs:
        if x == None:
            return False
    return True


class PatternHiter(object):
    '''
    模式识别器
    '''

    def __init__(self, location, mode=1):
        if mode == 1:
            if ':' in location:
                self.pps = dict()
                _init_patterns(location, self.pps)
            else:
                self.pps = init_patterns(location)
        else:
            if ':' in location:
                self.pps = dict()
                _init_patterns2(location, self.pps)
            else:
                self.pps = init_patterns2(location)

    def is_in(self, line):
        '''
        检查输入中是否命中规则
        :param line: str or unicode
        :return: 命中返回True
        '''
        if isinstance(line, str):
            line = unistr(line)
        if not isinstance(line, string_types):
            raise Exception("输入必须是str或unicode")
        for name, (ps1, ps2) in self.pps.items():
            t1 = regex(ps1, line)
            t2 = regex(ps2, line)
            if (len(ps1) == 0 or t1 != True) and t2 == True:  # 不满足1的同时满足2才行
                # print line.encode('utf-8'),name.encode('utf-8')
                return True
        return False

    def hit_num(self, line):
        '''
        计算输入命中规则的次数
        :param line: str or unicode
        :return: int, 命中次数
        '''
        if isinstance(line, str):
            line = unistr(line)
        if not isinstance(line, string_types):
            raise Exception("输入必须是str或unicode")
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
            if (len(ps1) == 0 or t1 != True) and t2 == True:  # 不满足1的同时满足2才行
                hit_group.append(name)
        return hit_group

        # def hit_num(self, line):
        #     return len(self.hit_group(line))


if __name__ == '__main__':
    p = PatternHiter("../../example/process_pattern_regex.exp")
    print(p.hit_num("世界和平"))
    print(p.is_in(u"世界和平"))
