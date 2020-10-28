# coding: utf-8
# 2020/10/28 @ tongshiwei

from longling import path_append


def _model_path(model_path=None,
                model_dir="./",
                model_prefix="",
                model_suffix="",
                model_name="model.lgb"
                ):
    if model_path is None:
        return path_append(model_dir, model_prefix + model_name + model_suffix, to_str=True)
    return model_path


def train(
        batch_size=5000,
        epoch_num=10,
        params=None,
        model_path=None,
        model_dir="./",
        model_prefix="",
        model_suffix="",
        model_name="model.lgb",
        *args,
        **kwargs
):
    model_path = _model_path(model_path, model_dir, model_prefix, model_suffix, model_name)
