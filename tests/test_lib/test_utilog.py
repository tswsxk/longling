# coding: utf-8
# 2019/11/28 @ tongshiwei

import json
from longling import path_append, config_logging


def test_root_logger():
    logger = config_logging(level="info", console_log_level="info")
    logger.info("test_log")


def test_log(logger):
    logger.info("test_log")


def test_json_log(log_root):
    json_log_path = path_append(log_root, "test_json_log.json", to_str=True)
    json_logger = config_logging(
        filename=json_log_path,
        file_format="%(message)s",
        logger="test.json",
        mode="w",
    )
    json_logger.info(json.dumps({"name": "test"}))
    with open(json_log_path) as f:
        assert "name" in json.loads(f.readline())
