# coding: utf-8
# create by tongshiwei on 2020-10-11

import pytest

from longling.ml import choice


def test_choice():
    obj = [0.1, 0.4]
    with pytest.raises(ValueError):
        choice(obj)

    obj = {1: 0.1, 2: 0.4}
    with pytest.raises(ValueError):
        choice(obj)

    with pytest.raises(TypeError):
        choice(1)
