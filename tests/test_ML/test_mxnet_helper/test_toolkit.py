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
    with pytest.raises(KeyError):
        get_lr_scheduler("multifactor", learning_rate=0.01, discount=0.1, update_epoch=5,
                         warmup_epoch=1, batches_per_epoch=10)

    with pytest.raises(ValueError):
        get_lr_scheduler("linear", lr=0.01, max_update=0)
