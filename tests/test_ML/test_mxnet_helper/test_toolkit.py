# coding: utf-8
# 2021/8/5 @ tongshiwei

import pytest
from longling.ML.MxnetHelper import net_initialize


def test_init_error():
    class Error(object):
        pass
    with pytest.raises(TypeError):
        net_initialize(None, None, Error())
