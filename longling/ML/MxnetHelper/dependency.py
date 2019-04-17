# coding: utf-8
# create by tongshiwei on 2019/4/16

import pathlib

from longling.requires_install import requirements2list

__all__ = ["__dependency__"]

__dependency__ = requirements2list(
    pathlib.PurePath(__file__).with_name("requirements")
)
