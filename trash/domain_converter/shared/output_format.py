# coding: utf-8
# created by tongshiwei on 17-11-8

import numpy as np


def strlist(data):
    return [str(d) for d in data]


def output_str_format(data, separator=","):
    return separator.join(strlist(data))


def output_numpy_format(data):
    return np.array(data).tolist()
