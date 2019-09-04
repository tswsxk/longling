# coding: utf-8
# created by tongshiwei on 18-2-5
import math
import warnings
from collections import OrderedDict
from contextlib import contextmanager

from longling.lib import ProgressMonitor, IterableMonitor
from longling.lib.stream import flush_print

try:
    NAN = math.nan
except (AttributeError, ImportError):
    NAN = float('nan')

__all__ = ["ConsoleProgressMonitor"]


class ConsoleProgressMonitor(ProgressMonitor):
    def __init__(self, indexes: (dict, OrderedDict), values: (dict, None) = None,
                 batch_num=NAN, end_epoch=NAN, output=True, silent=False, ):
        super(ConsoleProgressMonitor, self).__init__(
            ConsoleProgressMonitorPlayer(
                indexes=indexes,
                values=values,
                batch_num=batch_num,
                end_epoch=end_epoch,
                output=output,
                silent=silent,
            )
        )

    def __call__(self, iterator, epoch, *args, **kwargs):
        self.player.batch_start(epoch)
        return IterableMonitor(iterator, self.player, self.player.batch_end)


class ConsoleProgressMonitorPlayer(object):
    def __init__(self,
                 indexes: (dict, OrderedDict), values: (dict, None) = None,
                 batch_num=NAN, end_epoch=NAN, output=True, silent=False,
                 **kwargs):

        if values is not None:
            assert type(indexes) == type(values)

        if isinstance(indexes, dict):
            indexes = OrderedDict(indexes)

        if values is not None:
            for prefix, names in indexes.items():
                for name in names:
                    _ = values[prefix][name]

        self.indexes = indexes

        self.batch_num = batch_num
        self._batch_num = None
        self.end_epoch = end_epoch

        self.output = output

        info_header = "{:>5}| {:>7}" + " " * 5 \
                      + "{:>10}" + " " * 2 + "{:>10}" + " " * 5

        index_header = ""
        arguments = []

        for prefix, _indexes in self.indexes.items():
            __indexes = ["%s-%s" % (prefix, _index) for _index in _indexes]
            arguments += __indexes
            index_header += (" " * 2).join(
                ["{:>%s}" % min(len(index), 15) for index in __indexes]
            )

        self.output_formatter = info_header + index_header

        self.index_prefix = self.output_formatter.format("Epoch", "Total-E",
                                                         "Batch", "Total-B",
                                                         *arguments)

        self.epoch = NAN
        self.silent = silent

        self.values = values

    def __call__(self, batch_no, **kwargs):
        arguments = []

        for prefix, names in self.indexes.items():
            if prefix not in kwargs:
                ref = self.values[prefix]
            else:
                ref = kwargs[prefix]
            for name in names:
                arguments.append(ref[name])

        for prefix, name_value in kwargs.items():
            if prefix not in self.indexes:
                warnings.warn("detect unknown prefix: %s, all arguments will be ignored" % prefix)

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
