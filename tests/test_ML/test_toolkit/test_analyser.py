# coding: utf-8
# 2019/12/10 @ tongshiwei

from longling.ML.toolkit.analyser.cli import select_max, arg_select_max


def test_cli():
    arg_select_max("auc", "prf:1:f1", src="result.json")
    arg_select_max("auc", "prf:1:f1", src="result.json", with_all=True)
    arg_select_max("auc", "prf:1:f1", src="result.json", with_keys="Epoch;train_time")

    select_max("result.json", "auc", "prf:1:f1")
    select_max("result.json", "auc", "prf:1:f1", with_all=True)
    select_max("result.json", "auc", "prf:1:f1", with_keys="Epoch;train_time")
