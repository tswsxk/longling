# coding: utf-8
# 2020/3/4 @ tongshiwei

__all__ = ["STAGE_CANDIDATES", "OVERRIDE"]

from collections import OrderedDict

STAGE_CANDIDATES = OrderedDict({
    "build": {},
    "test": {},
    "review": {"on_stop": True, "only_master": 'n', "manual": 'n'},
    "docs": {"stage_image_port": '80', "only_master": 'y', "manual": 'y'},
    "production": {"only_master": 'y', "manual": 'y'},
})

OVERRIDE = None
