# coding: utf-8
# 2021/6/21 @ tongshiwei
from tqdm import tqdm
from longling import as_list
from copy import deepcopy
from .utils import POrderedDict

__all__ = ["ranking_report"]


def ranking_auc(ranked_label):
    """
    Examples
    --------
    >>> ranking_auc([1, 1, 0, 0, 0])
    1.0
    >>> ranking_auc([0, 1, 0, 1, 0])
    0.5
    >>> ranking_auc([1, 0, 1, 0, 0])
    0.8333333333333334
    >>> ranking_auc([0, 0, 0, 1, 1])
    0.0
    """
    pos_num = sum(ranked_label)
    neg_num = len(ranked_label) - sum(ranked_label)
    if pos_num * neg_num == 0:  # pragma: no cover
        return 1
    return sum(
        [len(ranked_label[i + 1:]) - sum(ranked_label[i + 1:]) for i, score in enumerate(ranked_label) if score == 1]
    ) / (pos_num * neg_num)


def ranking_report(y_true, y_pred, k: (int, list) = None, continuous=False, coerce="ignore", pad_pred=-100,
                   metrics=None, bottom=False, verbose=True) -> POrderedDict:
    r"""

    Parameters
    ----------
    y_true
    y_pred
    k
    continuous
    coerce
    pad_pred
    metrics
    bottom
    verbose

    Returns
    -------

    Examples
    --------
    >>> y_true = [[1, 0, 0], [0, 0, 1]]
    >>> y_pred = [[0.75, 0.5, 1], [1, 0.2, 0.1]]
    >>> ranking_report(y_true, y_pred)  # doctest: +NORMALIZE_WHITESPACE
           ndcg@k  precision@k  recall@k  f1@k  len@k  support@k
    1   1.000000     0.000000       0.0   0.0    1.0          2
    3   0.565465     0.333333       1.0   0.5    3.0          2
    5   0.565465     0.333333       1.0   0.5    3.0          2
    10  0.565465     0.333333       1.0   0.5    3.0          2
    auc: 0.250000	map: 0.416667	mrr: 0.416667	coverage_error: 2.500000	ranking_loss: 0.750000	len: 3.000000
    support: 2
    >>> ranking_report(y_true, y_pred, k=[1, 3, 5])  # doctest: +NORMALIZE_WHITESPACE
           ndcg@k  precision@k  recall@k  f1@k  len@k  support@k
    1   1.000000     0.000000       0.0   0.0    1.0          2
    3   0.565465     0.333333       1.0   0.5    3.0          2
    5   0.565465     0.333333       1.0   0.5    3.0          2
    auc: 0.250000	map: 0.416667	mrr: 0.416667	coverage_error: 2.500000	ranking_loss: 0.750000	len: 3.000000
    support: 2
    >>> ranking_report(y_true, y_pred, bottom=True)  # doctest: +NORMALIZE_WHITESPACE
              ndcg@k  precision@k  recall@k  f1@k  len@k  support@k  ndcg@k(B) \
    1   1.000000     0.000000       0.0   0.0    1.0          2   1.000000
    3   0.565465     0.333333       1.0   0.5    3.0          2   0.806574
    5   0.565465     0.333333       1.0   0.5    3.0          2   0.806574
    10  0.565465     0.333333       1.0   0.5    3.0          2   0.806574
    <BLANKLINE>
        precision@k(B)  recall@k(B)   f1@k(B)  len@k(B)  support@k(B)
    1         0.500000         0.25  0.333333       1.0             2
    3         0.666667         1.00  0.800000       3.0             2
    5         0.666667         1.00  0.800000       3.0             2
    10        0.666667         1.00  0.800000       3.0             2
    auc: 0.250000	map: 0.416667	mrr: 0.416667	coverage_error: 2.500000	ranking_loss: 0.750000	len: 3.000000
    support: 2	map(B): 0.708333	mrr(B): 0.750000
    >>> ranking_report(y_true, y_pred, bottom=True, metrics=["auc"])  # doctest: +NORMALIZE_WHITESPACE
    auc: 0.250000   len: 3.000000	support: 2
    >>> y_true = [[0.9, 0.7, 0.1], [0, 0.5, 1]]
    >>> y_pred = [[0.75, 0.5, 1], [1, 0.2, 0.1]]
    >>> ranking_report(y_true, y_pred, continuous=True)  # doctest: +NORMALIZE_WHITESPACE
          ndcg@k  len@k  support@k
    3   0.675647    3.0          2
    5   0.675647    3.0          2
    10  0.675647    3.0          2
    mrr: 0.750000	len: 3.000000	support: 2
    >>> y_true = [[1, 0], [0, 0, 1]]
    >>> y_pred = [[0.75, 0.5], [1, 0.2, 0.1]]
    >>> ranking_report(y_true, y_pred)  # doctest: +NORMALIZE_WHITESPACE
        ndcg@k  precision@k  recall@k      f1@k  len@k  support@k
    1     1.00     0.500000       0.5  0.500000    1.0          2
    3     0.75     0.416667       1.0  0.583333    2.5          2
    5     0.75     0.416667       1.0  0.583333    2.5          2
    10    0.75     0.416667       1.0  0.583333    2.5          2
    auc: 0.500000	map: 0.666667	mrr: 0.666667	coverage_error: 2.000000	ranking_loss: 0.500000	len: 2.500000
    support: 2
    >>> ranking_report(y_true, y_pred, coerce="abandon")  # doctest: +NORMALIZE_WHITESPACE
       ndcg@k  precision@k  recall@k  f1@k  len@k  support@k
    1     1.0     0.500000       0.5   0.5    1.0          2
    3     0.5     0.333333       1.0   0.5    3.0          1
    auc: 0.500000	map: 0.666667	mrr: 0.666667	coverage_error: 2.000000	ranking_loss: 0.500000	len: 2.500000
    support: 2
    >>> ranking_report(y_true, y_pred, coerce="padding")  # doctest: +NORMALIZE_WHITESPACE
        ndcg@k  precision@k  recall@k      f1@k  len@k  support@k
    1     1.00     0.500000       0.5  0.500000    1.0          2
    3     0.75     0.416667       1.0  0.583333    2.5          2
    5     0.75     0.416667       1.0  0.583333    2.5          2
    10    0.75     0.416667       1.0  0.583333    2.5          2
    auc: 0.500000	map: 0.666667	mrr: 0.666667	coverage_error: 2.000000	ranking_loss: 0.500000	len: 2.500000
    support: 2
    >>> ranking_report(y_true, y_pred, bottom=True)  # doctest: +NORMALIZE_WHITESPACE
        ndcg@k  precision@k  recall@k      f1@k  len@k  support@k  ndcg@k(B)  \
    1     1.00     0.500000       0.5  0.500000    1.0          2   1.000000
    3     0.75     0.416667       1.0  0.583333    2.5          2   0.846713
    5     0.75     0.416667       1.0  0.583333    2.5          2   0.846713
    10    0.75     0.416667       1.0  0.583333    2.5          2   0.846713
    <BLANKLINE>
        precision@k(B)  recall@k(B)   f1@k(B)  len@k(B)  support@k(B)
    1         0.500000          0.5  0.500000       1.0             2
    3         0.583333          1.0  0.733333       2.5             2
    5         0.583333          1.0  0.733333       2.5             2
    10        0.583333          1.0  0.733333       2.5             2
    auc: 0.500000	map: 0.666667	mrr: 0.666667	coverage_error: 2.000000	ranking_loss: 0.500000	len: 2.500000
    support: 2	map(B): 0.791667	mrr(B): 0.750000
    >>> ranking_report(y_true, y_pred, bottom=True, coerce="abandon")  # doctest: +NORMALIZE_WHITESPACE
       ndcg@k  precision@k  recall@k  f1@k  len@k  support@k  ndcg@k(B)  \
    1     1.0     0.500000       0.5   0.5    1.0          2   1.000000
    3     0.5     0.333333       1.0   0.5    3.0          1   0.693426
    <BLANKLINE>
       precision@k(B)  recall@k(B)  f1@k(B)  len@k(B)  support@k(B)
    1        0.500000          0.5      0.5       1.0             2
    3        0.666667          1.0      0.8       3.0             1
    auc: 0.500000	map: 0.666667	mrr: 0.666667	coverage_error: 2.000000	ranking_loss: 0.500000
    len: 2.500000	support: 2	map(B): 0.791667	mrr(B): 0.750000
    >>> ranking_report(y_true, y_pred, bottom=True, coerce="padding")  # doctest: +NORMALIZE_WHITESPACE
        ndcg@k  precision@k  recall@k      f1@k  len@k  support@k  ndcg@k(B)  \
    1     1.00     0.500000       0.5  0.500000    1.0          2   1.000000
    3     0.75     0.416667       1.0  0.583333    2.5          2   0.846713
    5     0.75     0.416667       1.0  0.583333    2.5          2   0.846713
    10    0.75     0.416667       1.0  0.583333    2.5          2   0.846713
    <BLANKLINE>
        precision@k(B)  recall@k(B)   f1@k(B)  len@k(B)  support@k(B)
    1             0.50          0.5  0.500000       1.0             2
    3             0.50          1.0  0.650000       3.0             2
    5             0.30          1.0  0.452381       5.0             2
    10            0.15          1.0  0.257576      10.0             2
    auc: 0.500000	map: 0.666667	mrr: 0.666667	coverage_error: 2.000000	ranking_loss: 0.500000	len: 2.500000
    support: 2	map(B): 0.791667	mrr(B): 0.750000
    """
    import numpy as np
    from collections import OrderedDict
    from sklearn.metrics import (
        label_ranking_average_precision_score,
        ndcg_score, label_ranking_loss, coverage_error
    )
    assert coerce in {"ignore", "abandon", "raise", "padding"}
    if metrics is None:
        metrics = ["mrr", "ndcg"]
        if continuous is False:
            metrics.extend(["auc", "map", "coverage_error", "ranking_loss", "precision", "recall", "f1"])
    metrics = set(metrics)

    if k is not None:
        k = as_list(k)
    else:
        if continuous is True:
            k = [3, 5, 10]
        else:
            k = [1, 3, 5, 10]

    results = {
        "auc": [],
        "map": [],
        "mrr": [],
        "coverage_error": [],
        "ranking_loss": [],
        "len": [],
        "support": [],
    }
    if bottom:
        results.update({
            "map(B)": [],
            "mrr(B)": [],
        })
    k_results = {}
    for _k in k:
        k_results[_k] = {
            "ndcg@k": [],
            "precision@k": [],
            "recall@k": [],
            "f1@k": [],
            "len@k": [],
            "support@k": [],
        }
        if bottom:
            k_results[_k].update({
                "ndcg@k(B)": [],
                "precision@k(B)": [],
                "recall@k(B)": [],
                "f1@k(B)": [],
                "len@k(B)": [],
                "support@k(B)": [],
            })
    suffix = [""]
    if bottom:
        suffix += ["(B)"]

    for label, pred in tqdm(zip(y_true, y_pred), "ranking metrics", disable=not verbose):
        if continuous is False and "map" in metrics:
            results["map"].append(label_ranking_average_precision_score([label], [pred]))
            if bottom:
                results["map(B)"].append(label_ranking_average_precision_score(
                    [(1 - np.asarray(label)).tolist()],
                    [(-np.asarray(pred)).tolist()]
                ))

        if len(label) > 1 and continuous is False:
            if "coverage_error" in metrics:
                results["coverage_error"].append(coverage_error([label], [pred]))
            if "ranking_loss" in metrics:
                results["ranking_loss"].append(label_ranking_loss([label], [pred]))

        results["len"].append(len(label))
        results["support"].append(1)
        label_pred = list(sorted(zip(label, pred), key=lambda x: x[1], reverse=True))
        sorted_label = list(zip(*label_pred))[0]
        if "auc" in metrics:
            results["auc"].append(ranking_auc(sorted_label))
        if "mrr" in metrics:
            try:
                results["mrr"].append(1 / (np.asarray(sorted_label).nonzero()[0][0] + 1))
            except IndexError:  # pragma: no cover
                pass
            try:
                if bottom:
                    results["mrr(B)"].append(1 / (np.asarray(sorted_label[::-1]).nonzero()[0][0] + 1))
            except IndexError:  # pragma: no cover
                pass

        if metrics & {"ndcg", "precision", "recall", "f1"}:
            for _k in k:
                for _suffix in suffix:
                    if _suffix == "":
                        _label_pred = deepcopy(label_pred)
                        if len(_label_pred) < _k:
                            if coerce == "ignore":
                                pass
                            elif coerce == "abandon":
                                continue
                            elif coerce == "raise":
                                raise ValueError("Not enough value: %s vs target %s" % (len(_label_pred), _k))
                            elif coerce == "padding":  # pragma: no cover
                                _label_pred += [(0, pad_pred)] * (_k - len(_label_pred))
                        k_label_pred = label_pred[:_k]
                        total_label = sum(label)
                    else:
                        inv_label_pred = [(1 - _l, -p) for _l, p in label_pred][::-1]
                        if len(inv_label_pred) < _k:
                            if coerce == "ignore":
                                pass
                            elif coerce == "abandon":
                                continue
                            elif coerce == "raise":  # pragma: no cover
                                raise ValueError("Not enough value: %s vs target %s" % (len(inv_label_pred), _k))
                            elif coerce == "padding":
                                inv_label_pred += [(0, pad_pred)] * (_k - len(inv_label_pred))
                        k_label_pred = inv_label_pred[:_k]
                        total_label = len(label) - sum(label)

                    if not k_label_pred:  # pragma: no cover
                        continue
                    k_label, k_pred = list(zip(*k_label_pred))
                    if "ndcg" in metrics:
                        if len(k_label) == 1 and "ndcg" in metrics:
                            k_results[_k]["ndcg@k%s" % _suffix].append(1)
                        else:
                            k_results[_k]["ndcg@k%s" % _suffix].append(ndcg_score([k_label], [k_pred]))
                    p = sum(k_label) / len(k_label)
                    r = sum(k_label) / total_label if total_label else 0
                    if "precision" in metrics:
                        k_results[_k]["precision@k%s" % _suffix].append(p)
                    if "recall" in metrics:
                        k_results[_k]["recall@k%s" % _suffix].append(r)
                    if "f1" in metrics:
                        k_results[_k]["f1@k%s" % _suffix].append(2 * p * r / (p + r) if p + r else 0)
                    k_results[_k]["len@k%s" % _suffix].append(len(k_label))
                    k_results[_k]["support@k%s" % _suffix].append(1)

    ret = POrderedDict()
    for key, value in results.items():
        if value:
            if key == "support":
                ret[key] = np.sum(value).item()
            else:
                ret[key] = np.mean(value).item()

    if metrics & {"ndcg", "precision", "recall", "f1"}:
        for k, key_value in k_results.items():
            ret[k] = OrderedDict()
            for key, value in key_value.items():
                if value:
                    if key in {"support@k", "support@k(B)"}:
                        ret[k][key] = np.sum(value).item()
                    else:
                        ret[k][key] = np.mean(value).item()
    return ret
