# coding: utf-8
# 2020/4/16 @ tongshiwei

from collections import OrderedDict

import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score

from longling import as_list


def regression_report(
        y_true, y_pred,
        metrics=None,
        sample_weight=None, multioutput="uniform_average", average_options=None,
        key_prefix="", key_suffix="",
        verbose=True,
):
    """

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    metrics: list of str,
        Support: evar(explained_variance), mse, rmse, mae, r2

    sample_weight : array-like of shape (n_samples,), optional
        Sample weights.

    multioutput : string in ['raw_values', 'uniform_average', 'variance_weighted'], list
        or array-like of shape (n_outputs)
        Defines aggregating of multiple output values.
        Disabled when verbose is True.
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
            Alias: "macro"
        ‘variance_weighted’ :
            Only support in evar and r2.
            Scores of all outputs are averaged, weighted by the variances of each individual output.
            Alias: "vw"

    average_options: str or list
        default to macro, choices (one or many): "macro", "vw"

    key_prefix: str
    key_suffix: str
    verbose: bool
    Returns
    -------

    """
    legal_metrics = ["evar", "rmse", "mse", "mae", "r2"]
    if not metrics:
        metrics = legal_metrics

    _metrics = set(metrics)
    assert not _metrics - set(legal_metrics)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    average_options = as_list(average_options) if average_options else ["macro", "vw"]

    alias_dict = {
        "macro": "uniform_average",
        "vw": "variance_weighted",
    }

    ret = OrderedDict()

    if len(y_true.shape) > 1 and verbose:
        _ret = regression_report(
            y_true, y_pred, sample_weight=sample_weight, metrics=_metrics,
            multioutput="raw_values",
            key_prefix=key_prefix, key_suffix=key_suffix, verbose=False,
        )
        for i in range(y_true.shape[1]):
            ret[i] = {}
            for _metric in _ret.keys():
                ret[i][_metric] = _ret[_metric][i]

        for _multioutput in average_options:
            __multioutput = _multioutput if isinstance(_multioutput, list) else alias_dict.get(_multioutput,
                                                                                               _multioutput)
            _ret = regression_report(
                y_true, y_pred, metrics=_metrics, sample_weight=sample_weight,
                multioutput=__multioutput, key_prefix=key_prefix, key_suffix=key_suffix, verbose=False
            )
            _name = "weighted" if isinstance(_multioutput, list) else __multioutput
            ret[_name] = {}
            for _metric in _ret:
                ret[_name][_metric] = _ret[_metric]

    else:
        if "evar" in _metrics:
            ret[key_prefix + "evar" + key_suffix] = explained_variance_score(y_true, y_pred,
                                                                             sample_weight=sample_weight,
                                                                             multioutput=multioutput)

        if "mse" in _metrics:
            _multioutput = "uniform_average" if multioutput == "variance_weighted" else multioutput
            ret[key_prefix + "mse" + key_suffix] = mean_squared_error(y_true, y_pred, sample_weight=sample_weight,
                                                                      multioutput=_multioutput)
        if "rmse" in _metrics:
            _multioutput = "uniform_average" if multioutput == "variance_weighted" else multioutput
            ret[key_prefix + "rmse" + key_suffix] = mean_squared_error(y_true, y_pred, sample_weight=sample_weight,
                                                                       squared=False,
                                                                       multioutput=_multioutput)

        if "mae" in _metrics:
            _multioutput = "uniform_average" if multioutput == "variance_weighted" else multioutput
            ret[key_prefix + "mae" + key_suffix] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight,
                                                                       multioutput=_multioutput)

        if "r2" in metrics:
            ret[key_prefix + "r2" + key_suffix] = r2_score(y_true, y_pred, sample_weight=sample_weight,
                                                           multioutput=multioutput)

    return ret


if __name__ == '__main__':
    from longling.ML.toolkit.formatter import EvalFMT

    # y_true = [3, -0.5, 2, 7]
    # y_pred = [2.5, 0.0, 2, 8]
    # y_true = [[0.5, 1], [-1, 1], [7, -6]]
    # y_pred = [[0, 2], [-1, 2], [8, -5]]
    y_true = [[0.5, 1, 1], [-1, 1, 1], [7, -6, 1]]
    y_pred = [[0, 2, 1], [-1, 2, 1], [8, -5, 1]]
    print(EvalFMT.format(
        eval_name_value=regression_report(y_true, y_pred, multioutput="variance_weighted", verbose=False))
    )
    print(EvalFMT.format(eval_name_value=regression_report(y_true, y_pred, verbose=False)))
    print(EvalFMT.format(eval_name_value=regression_report(y_true, y_pred, multioutput=[0.3, 0.6, 0.1]), verbose=False))
    print(EvalFMT.format(eval_name_value=regression_report(y_true, y_pred, verbose=True)))
    print(EvalFMT.format(
        eval_name_value=regression_report(
            y_true, y_pred, verbose=True, average_options=["macro", "vw", [0.3, 0.6, 0.1]]
        )
    ))
