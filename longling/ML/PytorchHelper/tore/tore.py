# coding: utf-8
# create by tongshiwei on 2019-9-4


from __future__ import absolute_import
from __future__ import print_function

import functools
import os

import longling
from longling.ML.DL import ecli
from longling.lib.parser import path_append

__all__ = ["cli"]

TORE_DIR = path_append(
    os.path.dirname(longling.__file__),
    "ML.PytorchHelper.tore.ModelName".replace(".", os.sep),
    to_str=True
)
cli = functools.partial(ecli, source_dir=TORE_DIR)

if __name__ == '__main__':
    cli()
