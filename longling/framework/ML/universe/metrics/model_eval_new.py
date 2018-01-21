# coding: utf-8
# create by tongshiwei on 2017/11/20
#
# def model_eval(predicts, golds, predict_probs):
#     num_correct = 0.
#     num_total = 0.
#     cat_res = evaluate(golds, predicts)
#     for i in xrange(len(predicts)):
#         if predicts[i] == golds[i]:
#             num_correct += 1.
#         num_total += 1.
#     dev_acc = num_correct * 100 / float(num_total)
#     if predict_probs is not None:
#         AUC = caculate_AUC(golds, predict_probs)
#         return dev_acc, cat_res, AUC
#     else:
#         return dev_acc, cat_res
#
#
# def output_eval_res(predicts, golds, predict_probs=None, f_log=None):
#     if predict_probs is None:
#         dev_acc, cat_res = model_eval(predicts, golds, predict_probs)
#     else:
#         dev_acc, cat_res, AUC = model_eval(predicts, golds, predict_probs)
#     if f_log is not None:
#         print >> f_log, '--- Dev Accuracy thus far: %.3f' % dev_acc
#     print '--- Dev Accuracy thus far: %.3f' % dev_acc
#     for cat, res in cat_res.items():
#         if f_log is not None:
#             print >> f_log, '--- Dev Category %s P=%s,R=%s,F=%s' % (cat, res[0], res[1], res[2])
#         print '--- Dev Category %s P=%s,R=%s,F=%s' % (cat, res[0], res[1], res[2])
#     if predict_probs is not None:
#         print >> f_log, '--- Dev AUC finally %s' % AUC
#         print '--- Dev AUC finally %s' % AUC