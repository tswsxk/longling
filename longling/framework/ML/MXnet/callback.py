# coding: utf-8
# created by tongshiwei on 18-1-24

from __future__ import print_function

import codecs
import logging
import json
import time

import mxnet as mx
import numpy as np

from tqdm import tqdm

from longling.lib.stream import build_dir


def extract2tuple(*listobj):
    return listobj


def as_list(obj):
    if isinstance(obj, (tuple, set, list)):
        return obj
    else:
        return [obj]


# these method are used as batch_end_callback method
# do not use vd methods recently
#######################################################################################
# from visualdl import LogWriter
# class VDLogger(object):
#     def __init__(self, logdir, mode_name, recorder, sync_cycle=10, recorder_params={}):
#         self.logdir = logdir
#         self.logger = LogWriter(logdir, sync_cycle=sync_cycle)
#         with self.logger.mode(mode_name):
#             self.recorder = eval("self.logger." + recorder)(**recorder_params)
#
#     def __call__(self, *args, **kwargs):
#         raise NotImplementedError
#
#
# class VDEvalLogger(VDLogger):
#     def __init__(self, logdir, mode_name, recorder="scalar", sync_cycle=10, recorder_params={}):
#         super(VDEvalLogger, self).__init__(logdir, mode_name, recorder, sync_cycle, recorder_params)
#         self.cnt_step = 0
#
#     def __call__(self, *args, **kwargs):
#         params = args[0]
#         for name, value in params.eval_metric.get_name_value():
#             self.recorder.add_record(self.cnt_step, value)
#         self.cnt_step += 1
#######################################################################################

#######################################################################################
# Reconstruction
def do_checkpoint(prefix, period=1):
    build_dir(prefix)
    return mx.callback.do_checkpoint(prefix, period)


class Speedometer(mx.callback.Speedometer):
    def __init__(self, batch_size, frequent=50, auto_reset=True, metrics=None, logger=logging):
        super(Speedometer, self).__init__(batch_size, frequent, auto_reset)
        self.metrics = None if metrics is None else as_list(metrics)
        self.logger = logger

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                msg = "Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec" % (param.epoch, count, speed)
                if self.metrics is not None:
                    for metric in self.metrics:
                        name_value = metric.get_name_value()
                        msg += '\t%s=%f' * len(name_value) % extract2tuple(*sum(name_value, ()))
                self.logger.info(msg)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()


#######################################################################################

#######################################################################################
# tqdm based

class TqdmSpeedometer(tqdm):
    def __call__(self, param):
        self.update()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def TqdmEpochReset(tqdm_ins, desc):
    def _callback(iter_no, sym, arg, aux):
        tqdm_ins.set_description_str("%s Epoch[%d]" % (desc, iter_no + 1), refresh=False)

    return _callback


def tqdm_speedometer(begin_epoch=0, total=None):
    return TqdmSpeedometer(desc="training Epoch[%d]" % begin_epoch, unit=" batches",
                           bar_format="{desc}: [{elapsed}<{remaining}, {rate_fmt}]", total=total)


#######################################################################################

#######################################################################################
# NEW

# eval epoch end callback
class ClassificationLogValidationMetricsCallback(object):
    """Just logs the eval metrics at the end of an epoch."""

    def __init__(self, time_monitor=None, loss_metrics=None, logger=logging, logfile=None):
        self.time_monitor = time_monitor
        self.tic = 0
        self.logger = logger
        self.log_f = None
        if logfile is not None:
            self.log_f = codecs.open(logfile, "w", "utf-8")

        self.loss_metrics = as_list(loss_metrics) if loss_metrics is not None and loss_metrics else None

    def __call__(self, param, msg_name_group=None):
        assert param.eval_metric is not None
        data = {}
        msg = 'Iter [%d]:' % param.epoch
        data['iteration'] = param.epoch
        if self.time_monitor is not None:
            train_time = self.time_monitor.total_wall_time - self.tic
            self.tic = self.time_monitor.total_wall_time
            msg += "\tTrain Time-%.3fs" % train_time
            data['train_time'] = train_time

        name_value = dict(param.eval_metric.get_name_value())
        if 'accuracy' in name_value:
            accuracy = name_value['accuracy']
            msg += "\tValidation Accuracy: %f" % name_value['accuracy']
            data['accuracy'] = accuracy

        del name_value['accuracy']

        if self.loss_metrics is not None:
            msg += "\tLoss - "
            for loss_metric in self.loss_metrics:
                n, v = loss_metric.get_name_value()[0]
                msg += "%s: %f" % (n, v)

        prf = {}
        eval_ids = set()
        for name, value in name_value.items():
            try:
                eval_id, class_id = name.split("_")
            except ValueError:
                continue
            if class_id not in prf:
                prf[class_id] = {}
            prf[class_id][eval_id] = value
            eval_ids.add(eval_id)

        if prf:
            avg = {eval_id: [] for eval_id in eval_ids}
            for class_id in sorted(prf.keys()):
                for eval_id, values in avg.items():
                    values.append(prf[class_id][eval_id])
                msg += "\n"
                msg += "--- Category %s " % class_id
                msg_res = sorted(prf[class_id].items(), reverse=True)
                msg += ("\t{}={:.10f}" * len(prf[class_id])).format(*sum(msg_res, ()))
            avg = {eval_id: np.average(values) for eval_id, values in avg.items()}
            msg += "\n"
            msg += "--- Category_A "
            msg_res = sorted(avg.items(), reverse=True)
            msg += ("\t{}={:.10f}" * len(avg)).format(*sum(msg_res, ()))
            prf['avg'] = avg
            data['prf'] = prf

        self.logger.info(msg)
        if self.log_f is not None:
            print(json.dumps(data, ensure_ascii=False), file=self.log_f)

# epoch end callback
#######################################################################################
