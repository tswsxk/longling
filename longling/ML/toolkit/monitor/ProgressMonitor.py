# coding: utf-8
# created by tongshiwei on 18-2-5
import math
from contextlib import contextmanager

from longling.lib.stream import flush_print

try:
    NAN = math.nan
except (AttributeError, ImportError):
    NAN = float('nan')

__all__ = ["ConsoleProgressMonitor"]


class ConsoleProgressMonitor(object):
    def __init__(self, loss_index=None, eval_index=None, batch_num=NAN,
                 end_epoch=NAN, output=True, silent=False,
                 **kwargs):
        loss_index = [] if loss_index is None else loss_index
        eval_index = [] if eval_index is None else eval_index

        self.batch_num = batch_num
        self._batch_num = None
        self.end_epoch = end_epoch

        self.eval_index = eval_index
        self.loss_index = ["Loss-%s" % ln for ln in loss_index]

        self.output = output

        info_header = "{:>5}| {:>7}" + " " * 5 \
                      + "{:>10}" + " " * 2 + "{:>10}" + " " * 5
        loss_header = (" " * 2).join(
            ["{:>%s}" % min(len(index), 15) for index in loss_index]
        )
        eval_header = (" " * 2).join(
            ["{:>%s}" % min(len(index), 15) for index in eval_index]
        )

        self.output_formatter = info_header + loss_header + eval_header

        arguments = list(self.loss_index) + list(self.eval_index)
        self.index_prefix = self.output_formatter.format("Epoch", "Total-E",
                                                         "Batch", "Total-B",
                                                         *arguments)

        self.epoch = NAN
        self.silent = silent

    def __call__(self, batch_no, loss_value=None, eval_value=None):
        loss_value = [] if loss_value is None else loss_value
        eval_value = [] if eval_value is None else eval_value

        arguments = list(loss_value) + list(eval_value)
        res_str = self.output_formatter.format(self.epoch, self.end_epoch,
                                               batch_no, self.batch_num,
                                               *arguments)

        if self.output and not self.silent:
            flush_print(res_str)

        self._batch_num = batch_no if self._batch_num is None else max(
            [batch_no, self._batch_num]
        )
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
        elif self._batch_num is not None:
            self.batch_num = self._batch_num
        if self.output and not self.silent:
            print("")

        return ""

    @contextmanager
    def watching(self, epoch):
        self.batch_start(epoch)
        yield
        self.batch_end()
