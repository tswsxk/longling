# coding: utf-8
# 2020/4/19 @ tongshiwei

from longling import Configuration
from longling.ML.toolkit.hyper_search import prepare_hyper_search


def test_hyper_search():
    cfg_kwargs, _, _, _ = prepare_hyper_search(
        {"learning_rate": 0.1}, primary_key="macro_avg:f1", with_keys="accuracy", disable=True
    )
    assert cfg_kwargs == {"learning_rate": 0.1}

    cfg_kwargs, reporthook, final_reporthook, dump = prepare_hyper_search(
        {"learning_rate": 0.1}, primary_key="macro_avg:f1", with_keys="accuracy"
    )

    assert cfg_kwargs == {'learning_rate': 0.1}

    reporthook({"macro_avg": {"f1": 0.5}, "accuracy": 0.7})
    reporthook({"macro_avg": {"f1": 0.5}, "accuracy": 0.6})

    final_reporthook()

    cfg_kwargs, reporthook, final_reporthook, dump = prepare_hyper_search(
        {"learning_rate": 0.1}, primary_key="macro_avg:f1", with_keys="accuracy;macro_avg:precision"
    )

    reporthook({"macro_avg": {"f1": 0.5, "precision": 0.6}, "accuracy": 0.7})
    reporthook({"macro_avg": {"f1": 0.5, "precision": 0.6}, "accuracy": 0.6})

    final_reporthook()

    cfg_kwargs, reporthook, final_reporthook, dump = prepare_hyper_search(
        {"learning_rate": 0.1}, primary_key="macro_avg:f1"
    )

    reporthook({"macro_avg": {"f1": 0.5, "precision": 0.6}, "accuracy": 0.7})
    reporthook({"macro_avg": {"f1": 0.5, "precision": 0.6}, "accuracy": 0.6})

    final_reporthook()
