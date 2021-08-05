# coding: utf-8
# 2021/8/4 @ tongshiwei

import random
import pytest

FEATURE_DIM = 10
SAMPLE_NUM = 500


@pytest.fixture(scope="module")
def constants():
    return FEATURE_DIM


@pytest.fixture(scope="module")
def data_x():
    return [[random.random() for _ in range(FEATURE_DIM)] for _ in range(SAMPLE_NUM)]


@pytest.fixture(scope="module")
def data_y():
    return [random.randint(0, 1) for _ in range(SAMPLE_NUM)]
