# coding: utf-8
# 2019/11/28 @ tongshiwei

from longling import path_append
from longling import NullLogRF, JsonLogRF


def test_null_log_rf():
    to_log = {"a": 123}

    rf = NullLogRF()
    rf.write(to_log)
    rf.log(to_log)
    rf.add(to_log)
    rf.dump(to_log)

    assert True


def test_json_log_rf(tmp_path):
    to_log = {"a": 123}

    rf = JsonLogRF(path_append(tmp_path, "log_rf.json", to_str=True))
    rf.log(to_log)

    rf = JsonLogRF(None)
    rf.log(to_log)

    assert True
