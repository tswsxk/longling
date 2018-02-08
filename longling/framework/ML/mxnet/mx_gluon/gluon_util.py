# coding: utf-8
# created by tongshiwei on 18-2-5

import math
import pickle

from longling.lib.stream import flush_print


class TrainBatchInfoer(object):
    def __init__(self, loss_index=[], eval_index=[], batch_num=math.nan, epoch_num=math.nan, output=True):
        self.batch_num = batch_num
        self.epoch_num = epoch_num

        self.eval_index = eval_index
        self.loss_index = ["Loss-%s" % ln for ln in loss_index]

        self.output = output

        self.output_formatter = "{:>5}| {:>5}" + " " * 5 + "{:>10}" + " " * 2 + "{:>10}" + " " * 5 \
                                + (" " * 2).join(["{:>%s}" % min(len(index), 15) for index in loss_index]) \
                                + (" " * 2).join(["{:>%s}" % min(len(index), 15) for index in eval_index])

        self.index_prefix = self.output_formatter.format("Epoch", "Total", "Batch", "Total",
                                                         *self.loss_index, *self.eval_index)

        self.epoch = math.nan

    def report(self, batch_no, loss_value=[], eval_value=[]):
        res_str = self.output_formatter.format(self.epoch, self.epoch_num,
                                               batch_no, self.batch_num,
                                               *loss_value,
                                               *eval_value)

        if self.output:
            flush_print(res_str)

        return res_str

    def batch_start(self, epoch):
        res_str = self.index_prefix
        self.epoch = epoch

        if self.output:
            print(res_str)

        return res_str

    def batch_end(self, batch_num=None):
        if batch_num is not None:
            self.batch_num = batch_num
        if self.output:
            print("")

        return ""

