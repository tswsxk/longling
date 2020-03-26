# coding: utf-8
# 2020/3/24 @ tongshiwei

import datetime


def get_current_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")
