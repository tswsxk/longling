# coding: utf-8
# 2020/1/10 @ tongshiwei

from heapq import nlargest
from longling.ML.toolkit.analyser import get_max
from longling import Configuration, path_append

import json
import sqlite3
import os


def show_top_k(k, exp_id=None, exp_dir=path_append(os.environ["HOME"], "nni/experiments"), show=True):
    if exp_id:
        exp_dir = path_append(exp_dir, exp_id)
    sqlite_db = path_append(exp_dir, "db", "nni.sqlite", to_str=True)
    print(sqlite_db)
    conn = sqlite3.connect(sqlite_db)
    c = conn.cursor()
    cursor = c.execute("select trialJobId, data from MetricData;")
    _ret = []
    top_k = nlargest(k, [row for row in cursor], key=lambda x: float(x[1]))
    trial_dir = path_append(exp_dir, "trials")
    for trial, result in top_k:
        with open(path_append(trial_dir, trial, "parameter.cfg")) as f:
            trial_params = json.load(f)["parameters"]
            _ret.append([trial, result, trial_params])
    conn.close()

    if show:
        for e in _ret:
            print(e)
    return _ret


class BaseReporter(object):
    def intermediate(self, data):
        raise NotImplementedError

    def final(self):
        raise NotImplementedError


def get_params(received_params: dict, cfg_cls: Configuration):
    cfg_params = {}
    u_params = {}
    for k, v in received_params:
        if k in cfg_cls.vars():
            cfg_params[k] = v
        else:
            u_params[k] = v
    return cfg_params, u_params


def prepare_hyper_search(cfg_kwargs: dict, cfg_cls: Configuration, reporthook=None, final_reporthook=None,
                         final_key=None, reporter_cls=None):
    try:
        from nni import get_next_parameter, report_intermediate_result, report_final_result

        class Reporter(object):
            def __init__(self):
                self.datas = []

            def intermediate(self, data):
                # report_intermediate_result(data)
                self.datas.append(data)

            def final(self):
                report_final_result(float(get_max(self.datas, final_key)[0][final_key]))

        rc = Reporter() if reporter_cls is None else reporter_cls
        reporthook = reporthook if reporthook is not None else rc.intermediate
        final_reporthook = final_reporthook if final_reporthook is not None else rc.final
        cfg_cls_params, hyper_params = get_params(get_next_parameter(), cfg_cls)
        cfg_kwargs.update(cfg_cls_params)
        cfg_kwargs["hyper_params"].update(hyper_params)
        return cfg_kwargs, reporthook, final_reporthook

    except ModuleNotFoundError:
        return cfg_kwargs, reporthook, final_reporthook
