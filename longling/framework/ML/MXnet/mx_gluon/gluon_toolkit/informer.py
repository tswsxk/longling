# coding: utf-8
# created by tongshiwei on 18-2-5

from longling.lib.stream import flush_print

try:
    from math import nan

    NAN = nan
except ImportError:
    NAN = float('nan')


class TrainBatchInformer(object):
    def __init__(self, loss_index=None, eval_index=None, batch_num=NAN, end_epoch=NAN, output=True, silent=False,
                 **kwargs):
        loss_index = [] if loss_index is None else loss_index
        eval_index = [] if eval_index is None else eval_index

        self.batch_num = batch_num
        self.end_epoch = end_epoch

        self.eval_index = eval_index
        self.loss_index = ["Loss-%s" % ln for ln in loss_index]

        self.output = output

        self.output_formatter = "{:>5}| {:>7}" + " " * 5 + "{:>10}" + " " * 2 + "{:>10}" + " " * 5 \
                                + (" " * 2).join(["{:>%s}" % min(len(index), 15) for index in loss_index]) \
                                + (" " * 2).join(["{:>%s}" % min(len(index), 15) for index in eval_index])

        arguments = list(self.loss_index) + list(self.eval_index)
        self.index_prefix = self.output_formatter.format("Epoch", "Total-E", "Batch", "Total-B",
                                                         *arguments)

        self.epoch = NAN
        self.silent = silent

    def batch_report(self, batch_no, loss_value=None, eval_value=None):
        loss_value = [] if loss_value is None else loss_value
        eval_value = [] if eval_value is None else eval_value

        arguments = list(loss_value) + list(eval_value)
        res_str = self.output_formatter.format(self.epoch, self.end_epoch,
                                               batch_no, self.batch_num,
                                               *arguments)

        if self.output and not self.silent:
            flush_print(res_str)

        return res_str

    def batch_start(self, epoch):
        res_str = self.index_prefix
        self.epoch = epoch

        if self.output and not self.silent:
            print(res_str)

        return res_str

    def batch_end(self, batch_num=None):
        if batch_num is not None:
            self.batch_num = batch_num
        if self.output:
            print("")

        return ""
