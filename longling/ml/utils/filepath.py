# coding: utf-8
# create by tongshiwei on 2020-11-17

from longling import path_append

__all__ = ["epoch_params_filename", "get_epoch_params_filepath", "params_filename", "get_params_filepath"]


def params_filename(model_name: str):
    """
    Examples
    --------
    >>> params_filename("CNN")
    'CNN.params'
    """
    return "%s.params" % model_name


def get_params_filepath(model_name: str, model_dir: str = "./"):
    """
    Examples
    --------
    >>> get_params_filepath("CNN")
    'CNN.params'
    """
    return path_append(model_dir, params_filename(model_name), to_str=True)


def epoch_params_filename(model_name: str, epoch: int):
    """
    Examples
    --------
    >>> epoch_params_filename("CNN", 10)
    'CNN-0010.params'
    """
    return "%s-%04d.params" % (model_name, epoch)


def get_epoch_params_filepath(model_name: str, epoch: int, model_dir: str = "./"):
    """
    Examples
    --------
    >>> get_epoch_params_filepath("CNN", 10)
    'CNN-0010.params'
    """
    return path_append(model_dir, epoch_params_filename(model_name, epoch), to_str=True)
