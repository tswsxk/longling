# coding: utf-8
# create by tongshiwei on 2017/12/2
from __future__ import absolute_import

string_types = str,
integer_types = int


def unistr(x):
    return x


def tostr(x):
    return str(x)

# py2 aborted
# string_types = basestring,
# integer_types = (int, long)
# unistr = lambda x: x.decode('utf-8')
# tostr = lambda x: unicode(x)
