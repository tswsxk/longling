# coding: utf-8
# 2019/12/3 @ tongshiwei

import pytest
from longling import config_logging
from longling import path_append


@pytest.fixture(scope="session")
def log_root(tmp_path_factory):
    return tmp_path_factory.mktemp("log")


@pytest.fixture(scope="session")
def logger():
    return config_logging(logger="test")


@pytest.fixture(scope="session")
def log_path(log_root):
    return path_append(log_root, "test.log", to_str=True)


@pytest.fixture(scope="session")
def file_logger(log_path):
    return config_logging(
        filename=log_path,
        logger="test.file"
    )


@pytest.fixture(scope="session")
def json_log_path(log_root):
    return path_append(log_root, "test.log.json", to_str=True)


@pytest.fixture(scope="session")
def json_logger(json_log_path):
    return config_logging(
        filename=json_log_path,
        file_format="%(message)s",
        logger="test.json"
    )
