# coding: utf-8
# 2020/3/31 @ tongshiwei

import pytest
from longling.spider import retry


def test_retry_wrapper():
    obj = [0]

    @retry()
    def demo_function():
        if obj[0] < 3:
            obj[0] += 1
            raise ConnectionResetError
        else:
            return True

    assert demo_function()

    obj = [0]

    @retry(3)
    def demo_function():
        if obj[0] < 3:
            obj[0] += 1
            raise ConnectionResetError
        else:
            return True

    with pytest.raises(ConnectionError):
        demo_function()
