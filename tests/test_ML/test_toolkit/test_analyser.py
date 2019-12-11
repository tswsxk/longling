# coding: utf-8
# 2019/12/10 @ tongshiwei

import json
from longling import path_append, wf_open
from longling.ML.toolkit.analyser.cli import select_max, arg_select_max

result_demo = [
    {"Epoch": 0, "train_time": 283.87022066116333, "SLMLoss": 0.27298363054701535, "auc": 0.6983898502363638,
     "prf": {"0": {"f1": 0.7365337834483444, "precision": 0.7158642386342801, "recall": 0.7584324230372059},
             "1": {"f1": 0.554243381265959, "precision": 0.5827092900035323, "recall": 0.528429111410084},
             "avg": {"recall": 0.643430767223645, "f1": 0.6453885823571517, "precision": 0.6492867643189062}}},
    {"Epoch": 1, "train_time": 247.69247245788574, "SLMLoss": 0.2261042852750925, "auc": 0.7648225872059642,
     "prf": {"0": {"f1": 0.7708840736183995, "precision": 0.7574905258171483, "recall": 0.7847597820196919},
             "1": {"f1": 0.6240245239017395, "precision": 0.6426716455911058, "recall": 0.6064289832788776},
             "avg": {"recall": 0.6955943826492847, "f1": 0.6974542987600695, "precision": 0.7000810857041271}}},
    {"Epoch": 2, "train_time": 276.0030241012573, "SLMLoss": 0.19859076581378765, "auc": 0.8109025055870918,
     "prf": {"0": {"f1": 0.8026803636198995, "precision": 0.7901002258051737, "recall": 0.8156675902542763},
             "1": {"f1": 0.6777260141158684, "precision": 0.6958208904860724, "recall": 0.6605484015632007},
             "avg": {"recall": 0.7381079959087384, "f1": 0.7402031888678839, "precision": 0.7429605581456231}}}
]


def test_cli(tmp_path):
    tmp_file = path_append(tmp_path, "result.json", to_str=True)
    with wf_open(tmp_file) as wf:
        for r in result_demo:
            print(json.dumps(r), file=wf)

    arg_select_max("auc", "prf:1:f1", src=tmp_file)
    arg_select_max("auc", "prf:1:f1", src=tmp_file, with_all=True)
    arg_select_max("auc", "prf:1:f1", src=tmp_file, with_keys="Epoch;train_time")

    select_max(tmp_file, "auc", "prf:1:f1")
    select_max(tmp_file, "auc", "prf:1:f1", with_all=True)
    select_max(tmp_file, "auc", "prf:1:f1", with_keys="Epoch;train_time")
