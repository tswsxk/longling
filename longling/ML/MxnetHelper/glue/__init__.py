# coding: utf-8
# create by tongshiwei on 2018/8/5

from __future__ import absolute_import

from .ModelName import ModelName as MetaModel
from .ModelName.ModelName.Module import Module as MetaModule, Configuration as MetaCfg
from .glue import cli
from .module import module_wrapper
