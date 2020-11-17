# coding: utf-8
# create by tongshiwei on 2020-11-17

from longling import path_append

__all__ = ["epoch_params_filename", "get_epoch_params_filepath"]


def epoch_params_filename(model_name: str, epoch: int):
    return "%s-%04d.parmas" % (model_name, epoch)


def get_epoch_params_filepath(model_name: str, epoch: int, model_dir: str = "./"):
    return path_append(model_dir, epoch_params_filename(model_name, epoch), to_str=True)
