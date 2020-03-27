# coding: utf-8
# 2020/3/24 @ tongshiwei

import datetime


def get_current_timestamp() -> str:  # pragma: no cover
    """
    Examples
    ---------
    .. code-block :: python

    get_current_timestamp()
    '20200327172235'

    """
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")
