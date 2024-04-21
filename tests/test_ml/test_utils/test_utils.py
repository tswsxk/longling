# coding: utf-8
# create by tongshiwei on 2020-10-11

import pytest

from longling.ml import choice


def test_choice():
    obj = [0.1, 0.4]
    with pytest.raises(ValueError, match="the sum of obj should be 1, now is .*"):
        choice(obj)

    obj = {1: 0.1, 2: 0.4}
    with pytest.raises(ValueError, match="the sum of obj should be 1, now is .*"):
        choice(obj)

    with pytest.raises(TypeError, match= "obj should be either list or dict, now is .*"):
        choice(1)
