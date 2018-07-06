#coding:utf-8

from collections import defaultdict

def accuracy(ps, ys):
    n_acc = 0.
    for i in xrange(len(ps)):
        if ps[i] == ys[i]:
            n_acc += 1.
    return n_acc / len(ps)


def _evaluate(gold_set, predict_set):
    if len(gold_set) == 0:
        print 'gold set is empty'
        return 0, 1, 0
        # raise ValueError('gold set cannot be empty')
    if len(predict_set) == 0:
        return 1, 0, 0
    n_common = len(gold_set & predict_set)
    precision = n_common * 1.0 / len(predict_set)
    recall = n_common * 1.0 / len(gold_set)
    if precision + recall == 0:
        return  precision, recall, 0
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
        #print '%s: precision=%s recall=%s f1=%s' % (cat, p, r, f)
        cat_res[cat] = (p, r, f)

    return cat_res
