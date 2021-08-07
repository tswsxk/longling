# coding: utf-8
# 2021/8/5 @ tongshiwei

import pytest
from longling.ML.MxnetHelper import net_initialize, get_lr_scheduler


def test_init_error():
    class Error(object):
        pass

    with pytest.raises(TypeError):
        net_initialize(None, None, Error())


def test_lr_scheduler_error():
    with pytest.raises(ValueError):
        get_lr_scheduler("linear", lr=0.01, max_update=0)
