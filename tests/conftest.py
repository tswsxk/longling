# coding: utf-8
# 2019/12/3 @ tongshiwei

import pytest
from longling import config_logging
from longling import path_append


@pytest.fixture(scope="module")
def logger():
    config_logging(

    )


@pytest.fixture(scope="module")
def file_logger(tmpdir):
    return config_logging(
        filename=path_append(tmpdir, "test.log", to_str=True),
    )


@pytest.fixture(scope="module")
def json_logger(tmpdir):
    return config_logging(
        filename=path_append(tmpdir, "test.log.json", to_str=True)
    )
