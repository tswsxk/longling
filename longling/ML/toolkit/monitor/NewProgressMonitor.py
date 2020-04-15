# coding: utf-8
# 2020/4/15 @ tongshiwei

# coding: utf-8
# 2020/4/11 @ tongshiwei

import functools
import time
from tqdm import tqdm as std_tqdm
from longling import print_time, flush_print
from longling.lib.progress import MonitorPlayer, AsyncMonitorPlayer
from longling.lib.candylib import format_byte_sizeof

from sklearn.metrics import classification_report

# with print_time("testing"):
#     for i in range(10000):
#         pass
#
# with print_time("testing"):
#     with tqdm() as t:
#         for i in tqdm(range(10000)):
#             t.update(i)
#
# with print_time("testing"):
#     for i in range(10000):
#         flush_print("%s | 10000" % i)
#
mp = MonitorPlayer()

# with print_time("testing"):
#     for i in range(1000000):
#         mp(i)
#



def copydoc(cls):
    def _fn(fn):
        if fn.__name__ in cls.__dict__:
            fn.__doc__ = cls.__dict__[fn.__name__].__doc__
        return fn

    return _fn


import math
import warnings
from collections import OrderedDict
from contextlib import contextmanager

from longling.lib import ProgressMonitor, IterableMIcing
from longling.lib.stream import flush_print

try:
    NAN = math.nan
except (AttributeError, ImportError):
    NAN = float('nan')


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

        info_header = "{:>5}| {:>7}" + " " * 3 + "{:>10}| {:>10}" + " " * 5

        index_header = ""
        arguments = []

        for prefix, _indexes in self.indexes.items():
            __indexes = ["%s-%s" % (prefix, _index) for _index in _indexes]
            arguments += __indexes
            index_header += (" " * 2).join(
                ["{:>%s}" % min(len(index), 15) for index in __indexes]
            )

        self.progress_header = " " * 5 + "{:^30}"

        self.output_formatter = info_header + index_header

        _header_formatter = self.output_formatter + self.progress_header
        self.index_prefix = _header_formatter.format("Epoch", "Total-E",
                                                     "Batch", "Total-B",
                                                     *arguments, "Progress")

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


@functools.wraps(std_tqdm.__init__)
def tqdm(*args, epoch, **kwargs):
    return TQDM(*args, epoch=epoch, **kwargs)


class TQDM(std_tqdm):
    def __init__(self, *args, epoch, values, **kwargs):
        self.head = ConsoleProgressMonitorPlayer(
            indexes={
                "Loss": ["hello", "world"]
            },
            values={
                "Loss": values
            }
        )
        self.epoch = epoch
        super(TQDM, self).__init__(ncols=len(self.head.index_prefix), *args, **kwargs)

    def __iter__(self):
        if not self.disable:
            self.write("%s" % self.head.index_prefix, file=self.fp)

        return super(TQDM, self).__iter__()

    @property
    def format_dict(self):
        """Public API for read-only member access."""
        kwargs = {"hello": 1, "world": 2}
        arguments = []

        for prefix, names in self.head.indexes.items():
            if prefix not in kwargs:
                ref = self.head.values[prefix]
            else:
                ref = kwargs[prefix]
            for name in names:
                arguments.append(ref[name])

        # for prefix, name_value in kwargs.items():
        #     if prefix not in self.head.indexes:
        #         warnings.warn("detect unknown prefix: %s, all arguments will be ignored" % prefix)

        if self.dynamic_ncols:
            self.ncols, self.nrows = self.dynamic_ncols(self.fp)
        ncols, nrows = self.ncols, self.nrows
        self.desc = self.head.output_formatter.format(self.epoch, 10, self.n, 1, *arguments)
        return dict(
            n=self.n, total=self.total,
            elapsed=self._time() - self.start_t
            if hasattr(self, 'start_t') else 0,
            ncols=ncols, nrows=nrows,
            prefix=self.desc, ascii=self.ascii, unit=self.unit,
            unit_scale=self.unit_scale,
            rate=1 / self.avg_time if self.avg_time else None,
            bar_format=self.bar_format, postfix=self.postfix,
            unit_divisor=self.unit_divisor)


# with print_time("testing"):
#     for i in range(10000):
#         pass
#
# with print_time("testing"):
#     for i in std_tqdm(range(100)):
#         time.sleep(0.05)


# def no_it():
#     for i in range(100):
#         yield i
#
# values = {"hello": 1, "world": 2}
# with print_time("testing"):
#     for i in tqdm(range(100), epoch=1, values=values,
#                   bar_format="{desc}" + " " * 5 + "{percentage:>3.0f}% [{elapsed}<{remaining}, {rate_fmt}{postfix}]"):
#         values["hello"] = i
#         time.sleep(0.01)
        #
        # mp = MonitorPlayer()
        #
        # with print_time("testing"):
        #     for i in range(100):
        #         mp(i)
        #         time.sleep(0.05)
        #
        # mp = AsyncMonitorPlayer()
        # with print_time("testing"):
        #     for i in range(100):
        #         mp(i)
        #         time.sleep(0.05)
        #     mp.reset()
