# coding: utf-8
# 2021/8/5 @ tongshiwei

import pytest
from longling.ML.MxnetHelper import getF


def test_type_error():
    with pytest.raises(TypeError):
        getF("error")
