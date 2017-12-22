import time
import sys
sys.path.insert(0,"/opt/tiger/nlp/text_env/env/lib/python2.7/site-packages/mxnet-0.9.5-py2.7.egg")
import mxnet as mx
class Msger(object):
    def __init__(self, model_prefix, logs=sys.stderr):
        self.tic = time.time()
        self.logs = logs
        self.prefix = model_prefix
        self.filelogs = open(model_prefix[:-3] + "model.log", "w")

    def time_reset(self):
        self.tic = time.time()

    def write_msg(self, msg, chg = True):
        if chg:
            print >> self.logs, msg
            self.filelogs.write(msg.encode('utf-8') + '\n')
        else:
            print >> self.logs, msg,
            self.filelogs.write(msg.encode('utf-8'))
        self.filelogs.flush()

    def close(self):
        self.filelogs.close()

class epochCallbacker(object):
    def __init__(self, msger, eval_metric, eval_data):
        self.msger = msger
        self.eval_metric = eval_metric
        self.eval_data = eval_data

    def __call__(self, epoch, symbol, arg_params, aux_params):
        mx.model.save_checkpoint(self.msger.prefix, epoch, symbol, arg_params, aux_params)
        print >> self.msger.logs, "Saved checkpoint to %s-%04d.params" % (self.msger.prefix, epoch)
        self.msger.write_msg("Iter [%d]" % (epoch), False)



class evalEndCallbacker(object):
    def __init__(self, msger):
        self.msger = msger
        self.total_correct_num = 0
        self.total_num = 0

    def __call__(self, param):
        if not param.eval_metric:
            return
        n_v = param.eval_metric.get_name_value()

        name_value = {}
        for name, value in n_v:
            name_value[name] = value
        self.msger.write_msg("Train: Time: % .3fs, Training Accuracy: %.3f" % (time.time() - self.msger.tic, name_value['accuracy'] * 100))
        self.msger.time_reset()
        self.total_correct_num += name_value['correct_num']
        self.total_num += name_value['size']
        dev_acc = float(self.total_correct_num) / self.total_num
        self.msger.write_msg('--- Dev Accuracy thus far: %.3f' % (dev_acc * 100))

        p0, r0 = name_value['P0'], name_value['R0']
        f0 = 2.0 * p0 * r0 / (p0 + r0) if p0 + r0 else 0
        self.msger.write_msg('--- Dev %s P=%s,R=%s,F=%s' % (0, p0, r0, f0))

        p1, r1 = name_value['P1'], name_value['R1']
        f1 = 2.0 * p1 * r1 / (p1 + r1) if p1 + r1 else 0
        self.msger.write_msg('--- Dev %s P=%s,R=%s,F=%s' % (1, p1, r1, f1))