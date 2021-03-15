# coding: utf-8
# 2021/3/14 @ tongshiwei

import functools


def eval_wrapper(_eval_f):
    @functools.wraps(_eval_f)
    def eval_f(_net, *args, **kwargs):
        _net.eval()
        ret = _eval_f(_net, *args, **kwargs)
        _net.train()
        return ret

    return eval_f
