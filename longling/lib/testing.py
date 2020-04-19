# coding: utf-8
# 2020/4/14 @ tongshiwei
from contextlib import contextmanager

__all__ = ["simulate_stdin"]


@contextmanager
def simulate_stdin(*inputs):
    """
    测试中模拟标准输入

    Parameters
    ----------
    inputs: list of str

    Examples
    --------
    >>> with simulate_stdin("12", "", "34"):
    ...     a = input()
    ...     b = input()
    ...     c = input()
    >>> a
    '12'
    >>> b
    ''
    >>> c
    '34'
    """
    from unittest.mock import patch
    from io import StringIO
    with patch('sys.stdin', StringIO("\n".join(inputs) + "\n")):
        yield
