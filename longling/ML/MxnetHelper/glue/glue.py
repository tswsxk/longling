# coding: utf-8
# create by tongshiwei on 2018/8/5

from __future__ import absolute_import
from __future__ import print_function

import functools
import os

import longling
from longling.ML.DL import ecli
from longling.lib.parser import path_append

__all__ = ["cli"]

GLUE_DIR = path_append(
    os.path.dirname(longling.__file__),
    "ML.MxnetHelper.glue.ModelName".replace(".", os.sep),
    to_str=True
)
cli = functools.partial(ecli, source_dir=GLUE_DIR)

if __name__ == '__main__':
    cli()
