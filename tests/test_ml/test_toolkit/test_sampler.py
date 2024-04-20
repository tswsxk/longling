# coding: utf-8
# 2021/6/29 @ tongshiwei

import pytest
from longling.ml.toolkit.dataset import TripletPairSampler


def test_sampler():
    with pytest.raises(TypeError):
        TripletPairSampler.rating2triplet([1, 2], "query", "key")
