# coding: utf-8
# 2019/12/20 @ tongshiwei

import pytest
from longling.ML.toolkit.hyper_parameters import extract_params_combinations


def test_extract_params_combinations():
    candidates = {"a": {"b": [0, 3], "c": [1, 2], "d": "$c"}, "d": "$a:b"}
    combinations = extract_params_combinations(candidates)
    assert len(list(combinations)) == 4

    for candidate in combinations:
        assert candidate["a"]["d"] == candidate["a"]["c"]
        assert candidate["d"] == candidate["a"]["b"]

    candidates.update({"f": "$e"})
    external = {"e": 123}

    combinations = extract_params_combinations(candidates, external)

    for candidate in combinations:
        assert candidate["f"] == external["e"]

    candidates = {"a": "$b"}
    with pytest.raises(KeyError):
        for _ in extract_params_combinations(candidates):
            pass
