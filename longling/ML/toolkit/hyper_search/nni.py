# coding: utf-8
# 2020/1/10 @ tongshiwei

import pathlib
import warnings
from heapq import nlargest
from longling.ML.toolkit.analyser import get_max, get_min, get_by_key, key_parser
from longling import Configuration, path_append
from longling import dict2pv, list2dict

import json
import sqlite3
import os


def _key(x):
    """
    Examples
    --------
    >>> _key(123)
    123.0
    >>> _key('{"default": 123}')
    123.0
    """
    try:
        return float(x)
    except ValueError:
        return float(json.loads(x)["default"])


def show(key, max_key=True, exp_id=None, res_dir="./",
         nni_dir=path_append(os.environ.get("HOME", "./"), "nni/experiments"),
         only_final=False,
         with_keys=None, with_all=False):  # pragma: no cover
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
        if max_key:
            value, appendix = get_max(
                trial_res, key, with_keys=with_keys, with_all=with_all, merge=False
            )
        else:
            value, appendix = get_min(
                trial_res, key, with_keys=with_keys, with_all=with_all, merge=False
            )
        if with_keys is not None or with_all is True:
            result.append([trial, trial_params, value, dict(appendix[key])])
        else:
            result.append([trial, trial_params, value])
    conn.close()
    result.sort(key=lambda x: float(x[2][key]), reverse=True)
    return result


def show_top_k(k, exp_id=None,
               exp_dir=path_append(os.environ.get("HOME", "./"), "nni/experiments")):  # pragma: no cover
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


def get_params(received_params: dict, cfg_cls: (Configuration, type(Configuration))):
    """
    Parameters
    ----------
    received_params: dict
        nni get_next_parameters() 得到的参数字典
    cfg_cls: Configuration
        配置文件实例

    Returns
    -------
    cfg_params: dict
        存在与 cfg 中 received_params 的更新值
    unk_params: dict
        不存在与 cfg 中 received_params 的更新值

    Examples
    --------
    >>> class CFG(Configuration):
    ...     hyper_params = {"hidden_num": 100}
    ...     learning_rate = 0.001
    >>> cfg = CFG()
    >>> get_params({"hidden_num": 50, "learning_rate": 0.1, "act": "relu"}, cfg)
    ({'hyper_params': {'hidden_num': 50}, 'learning_rate': 0.1}, {'act': 'relu'})
    """
    cfg_params = {}
    unk_params = {}

    path, _ = dict2pv(cfg_cls.vars())

    keys = {p[-1]: p for p in path}

    for k, v in received_params.items():
        if k in cfg_cls.vars():
            cfg_params[k] = v
        else:
            if k in keys:
                cfg_params.update(list2dict(keys[k], v))
            else:
                unk_params[k] = v

    return cfg_params, unk_params


def prepare_hyper_search(cfg_kwargs: dict, cfg_cls: (Configuration, type(Configuration)),
                         reporthook=None, final_reporthook=None,
                         primary_key=None, max_key=True, reporter_cls=None, with_keys: (list, str, None) = None,
                         dump=False, disable=False):
    """
    从 nni package 中获取超参，更新配置文件参数。当 nni 不可用或不是 nni 搜索模式时，参数将不会改变。

    ..code-block :: python

        cfg_kwargs, reporthook, final_reporthook, tag = prepare_hyper_search(
            cfg_kwargs, Configuration, reporthook, final_reporthook, primary_key="macro_avg:f1"
        )

        _cfg = Configuration(**cfg_kwargs)
        model = Model(_cfg)
        ...

        for epoch in range(_cfg.begin_epoch, _cfg.end_epoch):
            for batch_data in dataset:
                train_model(batch_data)

            data = evaluate_model()
            reporthook(data)

        final_reporthook()

    Parameters
    ----------
    cfg_kwargs: dict
        待传入cfg的参数
    cfg_cls: type(Configuration) or Configuration
        配置文件
    reporthook
    final_reporthook
    primary_key:
        评估模型用的主键,
        ``nni.report_intermediate_result`` 和 ``nni.report_final_result``中 ``metric`` 的 ``default``
    max_key
    reporter_cls
    with_keys: list or str
        其它要存储的 metric
    dump: bool
        为 True 时，会修改 配置文件 中 workspace 参数为 ``workspace/nni.get_experiment_id()/nni.get_trial_id()``
        使得 nni 的中间结果会被存储下来。

    Returns
    -------
    cfg_kwargs: dict
        插入了nni超参后的配置文件参数
    reporthook: function
        每个iteration结束后的回调函数，用来报告中间结果。
        默认 ``nni.report_intermediate_result``。
    final_reporthook:
        所有iteration结束后的回调函数，用来报告最终结果。
        默认 ``nni.report_final_result``
    dump: bool
        和传入参数保持一致

    Examples
    --------
    .. code-block :: python

        class CFG(Configuration):
            hyper_params = {"hidden_num": 100}
            learning_rate = 0.001
            workspace = ""

        cfg_kwargs, reporthook, final_reporthook, dump = prepare_hyper_search(
            {"learning_rate": 0.1}, CFG, primary_key="macro_avg:f1", with_keys="accuracy"
        )
        # cfg_kwargs: {'learning_rate': 0.1}

    when nni start (e.g., using ``nni create --config _config.yml``),
    suppose in ``_config.yml``:

    .. code-block: yml

        searchSpacePath: _search_space.json

    and in ``_search_space.json``

    .. code-block :: json

        {
            "hidden_num": {"_type": "choice", "_value": [500, 600, 700, 835, 900]},
        }

    one of the return cfg_kwargs is ``{'hyper_params': {'hidden_num': 50}, 'learning_rate': 0.1}``
    """
    if disable:
        return cfg_kwargs, None, None, None
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
                feed_dict = {
                    'default': float(get_by_key(data, key_parser(primary_key))),
                    primary_key: get_by_key(data, key_parser(primary_key))
                }
                for key in with_keys:
                    feed_dict[key] = get_by_key(data, key_parser(key))
                report_intermediate_result(feed_dict)
                self.datas.append(data)

            def final(self):
                best_fn = get_min if max_key is False else get_max
                _with_keys = (with_keys if with_keys else []) + [primary_key]
                final_result = best_fn(
                    self.datas, primary_key, with_keys=";".join(_with_keys), merge=False
                )
                feed_dict = {
                    'default': float(final_result[0][primary_key])
                }
                appendix_dict = dict(final_result[1][primary_key])
                for key in _with_keys:
                    feed_dict[key] = appendix_dict[key]
                report_final_result(feed_dict)

        rc = Reporter() if reporter_cls is None else reporter_cls
        reporthook = reporthook if reporthook is not None else rc.intermediate
        final_reporthook = final_reporthook if final_reporthook is not None else rc.final
        cfg_cls_params, hyper_params = get_params(get_next_parameter(), cfg_cls)
        using_nni_tag = True if cfg_cls_params or hyper_params else False
        cfg_kwargs.update(cfg_cls_params)
        if "hyper_params" in cfg_kwargs:  # pragma: no cover
            cfg_kwargs["hyper_params"].update(hyper_params)
        if using_nni_tag is True and dump is True:  # pragma: no cover
            cfg_kwargs["workspace"] = cfg_kwargs.get("workspace", "") + path_append(
                nni.get_experiment_id(), nni.get_trial_id(), to_str=True
            )
        return cfg_kwargs, reporthook, final_reporthook, dump

    except ModuleNotFoundError:  # pragma: no cover
        warnings.warn("nni package not found, skip")
        return cfg_kwargs, reporthook, final_reporthook, dump
