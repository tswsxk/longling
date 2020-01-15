# coding: utf-8
# 2020/1/10 @ tongshiwei

import warnings
from heapq import nlargest
from longling.ML.toolkit.analyser import get_max, get_by_key, key_parser
from longling import Configuration, path_append
from longling import dict2pv, list2dict

import json
import sqlite3
import os


def _key(x):
    try:
        return float(x)
    except ValueError:
        return float(json.loads(x)["default"])


def show_top_k(k, exp_id=None, exp_dir=path_append(os.environ["HOME"], "nni/experiments")):
    if exp_id:
        exp_dir = path_append(exp_dir, exp_id)
    sqlite_db = path_append(exp_dir, "db", "nni.sqlite", to_str=True)
    print(sqlite_db)
    conn = sqlite3.connect(sqlite_db)
    c = conn.cursor()
    cursor = c.execute("select trialJobId, data from MetricData where type='FINAL';")
    _ret = []
    top_k = nlargest(k, [row for row in cursor], key=lambda x: _key(x[1]))
    trial_dir = path_append(exp_dir, "trials")
    for trial, result in sorted(top_k, key=lambda x: _key(x[1]), reverse=True):
        with open(path_append(trial_dir, trial, "parameter.cfg")) as f:
            trial_params = json.load(f)["parameters"]
            _ret.append([trial, result, trial_params])
    conn.close()

    return _ret


class BaseReporter(object):
    def intermediate(self, data):
        raise NotImplementedError

    def final(self):
        raise NotImplementedError


def get_params(received_params: dict, cfg_cls: Configuration):
    cfg_params = {}
    u_params = {}

    path, _ = dict2pv(cfg_cls.vars())

    keys = {p[-1]: p for p in path}

    for k, v in received_params.items():
        if k in cfg_cls.vars():
            cfg_params[k] = v
        else:
            if k in keys:
                cfg_params.update(list2dict(keys[k], v))
            else:
                u_params[k] = v

    return cfg_params, u_params


def prepare_hyper_search(cfg_kwargs: dict, cfg_cls, reporthook=None, final_reporthook=None,
                         primary_key=None, reporter_cls=None, with_keys: (list, str, None) = None):
    try:
        from nni import get_next_parameter, report_intermediate_result, report_final_result

        assert primary_key is not None

        if isinstance(with_keys, str):
            if ";" in with_keys:
                with_keys = with_keys.split(";")
            else:
                with_keys = [with_keys]
        elif with_keys is None:
            with_keys = []

        class Reporter(BaseReporter):
            def __init__(self):
                self.datas = []

            def intermediate(self, data):
                feed_dict = {'default': float(data[primary_key])}
                for key in with_keys:
                    feed_dict[key] = get_by_key(data, key_parser(key))
                report_intermediate_result(feed_dict)
                self.datas.append(data)

            def final(self):
                final_result = get_max(self.datas, primary_key, with_keys=";".join(with_keys) if with_keys else None)
                feed_dict = {
                    'default': float(final_result[0][primary_key])
                }
                appendix_dict = dict(final_result[1][primary_key])
                for key in with_keys:
                    feed_dict[key] = appendix_dict[key]
                report_final_result(feed_dict)

        rc = Reporter() if reporter_cls is None else reporter_cls
        reporthook = reporthook if reporthook is not None else rc.intermediate
        final_reporthook = final_reporthook if final_reporthook is not None else rc.final
        cfg_cls_params, hyper_params = get_params(get_next_parameter(), cfg_cls)
        using_nni_tag = cfg_cls_params or hyper_params
        cfg_kwargs.update(cfg_cls_params)
        cfg_kwargs["hyper_params"].update(hyper_params)
        return cfg_kwargs, reporthook, final_reporthook, using_nni_tag

    except ModuleNotFoundError:
        warnings.warn("nni package not found, skip")
        return cfg_kwargs, reporthook, final_reporthook, False
