# coding: utf-8
# 2020/4/16 @ tongshiwei

from collections import OrderedDict
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score


def regression_report(y_true, y_pred, metrics=None, sample_weight=None, multioutput="uniform_average", key_prefix=""):
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
        Array-like value defines weights used to average errors.
        'raw_values' :
            Returns a full set of errors in case of multioutput input.
        'uniform_average' :
            Errors of all outputs are averaged with uniform weight.
        ‘variance_weighted’ :
            Only support in evar and r2.
            Scores of all outputs are averaged, weighted by the variances of each individual output.

    Returns
    -------

    """
    if not metrics:
        metrics = [
            "evar", "mse", "mae", "r2"
        ]
    _metrics = set(metrics)

    ret = OrderedDict()

    if "evar" in _metrics:
        ret[key_prefix + "evar"] = explained_variance_score(y_true, y_pred, sample_weight=sample_weight,
                                                            multioutput=multioutput)

    if "mse" in _metrics:
        _multioutput = "uniform_average" if multioutput == "variance_weighted" else multioutput
        ret[key_prefix + "mse"] = mean_squared_error(y_true, y_pred, sample_weight=sample_weight,
                                                     multioutput=_multioutput)
    if "rmse" in _metrics:
        _multioutput = "uniform_average" if multioutput == "variance_weighted" else multioutput
        ret[key_prefix + "rmse"] = mean_squared_error(y_true, y_pred, sample_weight=sample_weight, squared=False,
                                                      multioutput=_multioutput)

    if "mae" in _metrics:
        _multioutput = "uniform_average" if multioutput == "variance_weighted" else multioutput
        ret[key_prefix + "mae"] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight,
                                                      multioutput=_multioutput)

    if "r2" in metrics:
        ret[key_prefix + "r2"] = r2_score(y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput)

    return ret


if __name__ == '__main__':
    from longling.ML.toolkit.formatter import EvalFMT

    # y_true = [3, -0.5, 2, 7]
    # y_pred = [2.5, 0.0, 2, 8]
    # y_true = [[0.5, 1], [-1, 1], [7, -6]]
    # y_pred = [[0, 2], [-1, 2], [8, -5]]
    y_true = [[0.5, 1], [-1, 1], [7, -6]]
    y_pred = [[0, 2], [-1, 2], [8, -5]]
    print(EvalFMT.format(eval_name_value=regression_report(y_true, y_pred, multioutput="variance_weighted")))
    print(EvalFMT.format(eval_name_value=regression_report(y_true, y_pred)))
    print(EvalFMT.format(eval_name_value=regression_report(y_true, y_pred, multioutput=[0.3, 0.7])))
