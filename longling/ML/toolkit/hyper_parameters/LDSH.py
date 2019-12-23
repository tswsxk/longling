# coding: utf-8
# 2019/12/20 @ tongshiwei

"""
Liu Ding Shen Huo
"""
from ..analyser.select import get_max
from .hyper_search import extract_params_combinations


def get_validation_file(model):
    return model.mod.cfg.validation_result_file


class LDSH(object):
    def __init__(self, model, metric_key, *positional_params, fix_params=None):
        attributes = ["train"]
        for attr in attributes:
            assert hasattr(model, attr)

        self.elixir = model
        self.fix_params = fix_params if fix_params else {}
        self.positional_params = positional_params
        self.metric_key = metric_key

    def search(self, candidates):
        for params in extract_params_combinations(candidates, self.fix_params):
            _params = {}
            _params.update(params)
            _params.update(self.fix_params)
            module = self.elixir.train(
                *self.positional_params,
                **_params
            )
            self.get_metric()

    def get_metric(self):
        result, _ = get_max(get_validation_file(self.elixir), self.metric_key, )
        return result[self.metric_key]
