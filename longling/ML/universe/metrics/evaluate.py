# coding:utf-8

from __future__ import print_function

from collections import defaultdict

def caculate_AUC(gold_list, predict_probs_list, punish_func=lambda x: 1):
    negative_cnt = 0
    positive_cnt = 0
    sorted_prob_gold = _plot_AUC(gold_list, predict_probs_list)
    current_score = sorted_prob_gold[0][0]
    equal_positive_cnt = 0
    punishment = 0
    for prob_gold in sorted_prob_gold:
        prob, gold = prob_gold
        if prob == current_score:
            equal_positive_cnt += 1 if gold == 1 else 0
        else:
            current_score = prob
            equal_positive_cnt = 0
        if gold == 1:
            positive_cnt += 1
            punishment += negative_cnt * punish_func(1)
        elif gold == 0:
            negative_cnt += 1
            punishment += equal_positive_cnt * 0.5 * punish_func(1)
        else:
            raise Exception("undefined gold label %s" % prob_gold)

    return 1 - punishment / (positive_cnt * negative_cnt)

def _plot_AUC(gold_list, predict_probs_list):
    assert len(gold_list) == len(predict_probs_list)
    prob_gold = zip(predict_probs_list, gold_list)
    prob_gold.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return prob_gold

def accuracy(ps, ys):
    n_acc = 0.
    for i in xrange(len(ps)):
        if ps[i] == ys[i]:
            n_acc += 1.
    return n_acc / len(ps)


def _evaluate(gold_set, predict_set):
    if len(gold_set) == 0:
        print('gold set is empty')
        return 0, 1, 0
        # raise ValueError('gold set cannot be empty')
    if len(predict_set) == 0:
        return 1, 0, 0
    n_common = len(gold_set & predict_set)
    precision = n_common * 1.0 / len(predict_set)
    recall = n_common * 1.0 / len(gold_set)
    if precision + recall == 0:
        return precision, recall, 0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate(gold_list, predict_list):
    cat_gold = defaultdict(set)
    cat_pred = defaultdict(set)
    for i, (g, p) in enumerate(zip(gold_list, predict_list)):
        if g not in cat_gold:
            cat_gold[g] = set()
        if p not in cat_pred:
            cat_pred[p] = set()
        cat_gold[g].add(i)
        cat_pred[p].add(i)

    cat_res = dict()
    for cat in cat_gold.keys():
        p, r, f = _evaluate(cat_gold[cat], cat_pred[cat])
        # print '%s: precision=%s recall=%s f1=%s' % (cat, p, r, f)
        cat_res[cat] = (p, r, f)

    return cat_res

if __name__ == '__main__':
    a = [
        1,
        0.9,
        0.5,
        0.3,
        0.95,
        0.18,
        0.34,
        0.67,
        0.88887,
        0.45,
    ]
    b = [
        1,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
    ]
    print(caculate_AUC(b, a))
