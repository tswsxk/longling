# coding: utf-8
# 2021/8/5 @ tongshiwei

import pytest
from longling.ML.PytorchHelper import get_optimizer_cfg


def test_key_error():
    with pytest.raises(KeyError):
        get_optimizer_cfg("error")
