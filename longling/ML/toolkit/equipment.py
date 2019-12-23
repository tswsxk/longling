# coding: utf-8
# 2019/12/20 @ tongshiwei

"""
mem_info.total: 显卡总的显存大小
mem_info.used: 使用显存大小
mem_info.free: 剩余显存大小

上述三个量单位都是字节bytes
"""

import pynvml

pynvml.nvmlInit()


def _get_device_info(device_id):
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    return mem_info


def _get_device_util(device_id):
    mem_info = _get_device_info(device_id)

    return mem_info.used / mem_info.total


def get_device_util(device_id=None):
    if device_id is not None:
        return _get_device_util(device_id)
    else:
        return [
            _get_device_util(i) for i in range(pynvml.nvmlDeviceGetCount())
        ]


def _get_device_free(device_id):
    mem_info = _get_device_info(device_id)

    return mem_info.free / mem_info.total


def get_device_free(device_id=None):
    if device_id is not None:
        return _get_device_free(device_id)
    else:
        return [
            _get_device_free(i) for i in range(pynvml.nvmlDeviceGetCount())
        ]


def get_free_device_ids(threshold=0.5):
    return [_id for _id, _f in get_device_free() if _f >= threshold]
