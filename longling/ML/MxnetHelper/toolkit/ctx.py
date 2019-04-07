# coding: utf-8
# create by tongshiwei on 2019/4/7
from mxnet.gluon import utils as gutil

from longling.lib.candylib import as_list

__all__ = ["real_ctx", "split_and_load"]


def real_ctx(ctx, data_len):
    ctx = as_list(ctx)
    if data_len < len(ctx):
        ctx = ctx[:1]
    return ctx


def split_and_load(ctx, *args, **kwargs):
    ctx = real_ctx(ctx, len(args[0]))
    return zip(*[gutil.split_and_load(arg, ctx, **kwargs) for arg in args])
