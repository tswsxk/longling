# coding: utf-8
# create by tongshiwei on 2017/10/9

from __future__ import absolute_import
from __future__ import print_function

from .evaluate import evaluate, caculate_AUC, _plot_AUC


def plot_ROC(predict_probs, golds, title="", savepath="", imshow=False):
    import matplotlib.pyplot as plt
    AUC = caculate_AUC(golds, predict_probs)
    if title:
        plt.figure(title)
        plt.title(title + "_ROC[AUC=%s]" % AUC)
    else:
        plt.figure()

    plt.xlabel(u"False Positive Rate, FPR")
    plt.ylabel(u"True Positive Rate, TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    sorted_probs_golds = _plot_AUC(golds, predict_probs)
    pos_cnt = len(filter(lambda x: x == 1, golds))
    neg_cnt = len(filter(lambda x: x == 0, golds))
    pos_factor, neg_factor = 1 / float(pos_cnt), 1 / float(neg_cnt)
    x = [0]
    y = [0]
    for i, prob_gold in enumerate(sorted_probs_golds):
        prob, gold = prob_gold
        new_x = x[i] if gold == 1 else x[i] + neg_factor
        new_y = y[i] + pos_factor if gold == 1 else y[i]
        x.append(new_x)
        y.append(new_y)
    plt.plot(x, y, color='black')
    plt.fill_between(x, y)
    plt.legend()
    if savepath:
        plt.savefig(savepath)
    elif imshow:
        plt.show()


def model_eval(predicts, golds, predict_probs):
    num_correct = 0.
    num_total = 0.
    cat_res = evaluate(golds, predicts)
    for i in xrange(len(predicts)):
        if predicts[i] == golds[i]:
            num_correct += 1.
        num_total += 1.
    dev_acc = num_correct * 100 / float(num_total)
    if predict_probs is not None:
        AUC = caculate_AUC(golds, predict_probs)
        return dev_acc, cat_res, AUC
    else:
        return dev_acc, cat_res


def output_eval_res(predicts, golds, predict_probs=None, f_log=None):
    if predict_probs is None:
        dev_acc, cat_res = model_eval(predicts, golds, predict_probs)
    else:
        dev_acc, cat_res, AUC = model_eval(predicts, golds, predict_probs)
    if f_log is not None:
        print('--- Dev Accuracy thus far: %.3f' % dev_acc, file=f_log)
    print('--- Dev Accuracy thus far: %.3f' % dev_acc)
    for cat, res in cat_res.items():
        if f_log is not None:
            print('--- Dev Category %s P=%s,R=%s,F=%s' % (cat, res[0], res[1], res[2]), file=f_log)
        print('--- Dev Category %s P=%s,R=%s,F=%s' % (cat, res[0], res[1], res[2]))
    if predict_probs is not None:
        print('--- Dev AUC finally %s' % AUC, file=f_log)
        print('--- Dev AUC finally %s' % AUC)


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
        0.1,
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
    plot_ROC(a, b)
    print(model_eval(a, b))
