# coding: utf-8
# 2020/1/10 @ tongshiwei

import pathlib
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


def show(key, exp_id=None, res_dir="./", nni_dir=path_append(os.environ.get("HOME", "./"), "nni/experiments"),
         only_final=False,
         with_keys=None, with_all=False):
    """
    cli alias: ``nni show``

    Parameters
    ----------
    key
    exp_id
    res_dir
    nni_dir
    only_final
    with_keys
    with_all

    Returns
    -------

    """
    if exp_id is None:
        exp_id = pathlib.PurePath(os.path.abspath(res_dir)).name
    nni_dir = path_append(nni_dir, exp_id)
    sqlite_db = path_append(nni_dir, "db", "nni.sqlite", to_str=True)
    print(sqlite_db)
    conn = sqlite3.connect(sqlite_db)
    c = conn.cursor()
    if only_final:
        cursor = c.execute("SELECT trialJobId FROM MetricData WHERE type='FINAL';")
    else:
        cursor = c.execute("SELECT DISTINCT trialJobId FROM MetricData;")
    trial_dir = path_append(nni_dir, "trials")
    result = []
    for trial in [row[0] for row in cursor]:
        with open(path_append(trial_dir, trial, "parameter.cfg")) as f:
            trial_params = json.load(f)["parameters"]
        trial_res = path_append(res_dir, trial, "result.json", to_str=True)
        value, appendix = get_max(trial_res, key, with_keys=with_keys, with_all=with_all)
        if with_keys is not None or with_all is True:
            result.append([trial, trial_params, value, dict(appendix[key])])
        else:
            result.append([trial, trial_params, value])
    conn.close()
    result.sort(key=lambda x: float(x[2][key]), reverse=True)
    return result


def show_top_k(k, exp_id=None, exp_dir=path_append(os.environ.get("HOME", "./"), "nni/experiments")):
    """
    cli alias: ``nni k-best``

    Parameters
    ----------
    k
    exp_id
    exp_dir

    Returns
    -------

    """
    if exp_id:
        exp_dir = path_append(exp_dir, exp_id)
    sqlite_db = path_append(exp_dir, "db", "nni.sqlite", to_str=True)
    print(sqlite_db)
    conn = sqlite3.connect(sqlite_db)
    c = conn.cursor()
    cursor = c.execute("SELECT trialJobId, data FROM MetricData WHERE type='FINAL';")
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
                         primary_key=None, reporter_cls=None, with_keys: (list, str, None) = None, dump=False):
    try:
        import nni
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
        using_nni_tag = True if cfg_cls_params or hyper_params else False
        cfg_kwargs.update(cfg_cls_params)
        cfg_kwargs["hyper_params"].update(hyper_params)
        if using_nni_tag is True and dump is True:
            cfg_kwargs["workspace"] = cfg_kwargs.get("workspace", "") + path_append(
                nni.get_experiment_id(), nni.get_trial_id(), to_str=True
            )
        return cfg_kwargs, reporthook, final_reporthook, dump

    except ModuleNotFoundError:
        warnings.warn("nni package not found, skip")
        return cfg_kwargs, reporthook, final_reporthook, dump
