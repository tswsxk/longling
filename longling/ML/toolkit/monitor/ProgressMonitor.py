# coding: utf-8
# created by tongshiwei on 18-2-5
import math
import sys
import warnings
from collections import OrderedDict
from contextlib import contextmanager

from longling import Timer
from longling.lib import ProgressMonitor, IterableMIcing
from longling.lib.stream import flush_print

try:
    NAN = math.nan
except (AttributeError, ImportError):  # pragma: no cover
    NAN = float('nan')

__all__ = ["ConsoleProgressMonitor", "ConsoleProgressMonitorPlayer"]


class ConsoleProgressMonitorPlayer(object):
    def __init__(self, indexes: (dict, OrderedDict), values: (dict, None) = None,
                 total=NAN, silent=False, timer=None, stream=sys.stdout, *args, **kwargs):
        """

        Parameters
        ----------
        indexes: dict
            {prefix: list of names}
        values: dict
            {prefix: list of functions}
        total
        silent
        timer
        stream
        args
        kwargs
        """

        if values is not None:
            assert type(indexes) == type(values)

        if isinstance(indexes, dict):
            indexes = OrderedDict(indexes)

        if values is not None:
            for prefix, names in indexes.items():
                for name in names:
                    _ = values[prefix][name]

        self.indexes = indexes

        self.n = None
        self.total = total
        self._total = None

        self.stream = stream
        self.timer = timer if timer is not None else Timer()

        index_header = ""
        arguments = []

        for prefix, _indexes in self.indexes.items():
            __indexes = ["%s-%s" % (prefix, _index) for _index in _indexes]
            arguments += __indexes
            index_header += (" " * 2).join(
                ["{:>%s}" % max(len(index), 15) for index in __indexes]
            )

        self.output_formatter = self.info_header + " " * 2 + index_header + " " * 2 + self.progress_header_fmt

        self.index_prefix = self.output_formatter.format(*self.headers, *arguments, self.progress_header)

        self.silent = silent
        self.values = values

    @property
    def progress_header_fmt(self):
        return "{:^30}"

    @property
    def progress_header(self):
        return "Progress"

    @property
    def progress(self):
        elapsed = self.timer.time()
        if not round(elapsed, 6) or not self.n:  # pragma: no cover
            return ""
        rate = self.n / elapsed
        elapsed = self.format_interval(int(elapsed))
        if self.total is not NAN:
            remaining = int((self.total - self.n) / rate)
            remaining = "<" + self.format_interval(remaining)
        else:
            remaining = ""

        return "[%s]" % (elapsed + remaining + ", {:.2f}it/s".format(rate))

    @staticmethod
    def format_interval(t):
        """
        Formats a number of seconds as a clock time, [H:]MM:SS

        Parameters
        ----------
        t  : int
            Number of seconds.

        Returns
        -------
        out  : str
            [H:]MM:SS

        Examples
        --------
        >>> ConsoleProgressMonitorPlayer.format_interval(360)
        '06:00'
        >>> ConsoleProgressMonitorPlayer.format_interval(100)
        '01:40'
        >>> ConsoleProgressMonitorPlayer.format_interval(36)
        '00:36'
        >>> ConsoleProgressMonitorPlayer.format_interval(3600)
        '1:00:00'
        >>> ConsoleProgressMonitorPlayer.format_interval(36000)
        '10:00:00'
        >>> ConsoleProgressMonitorPlayer.format_interval(360000)
        '100:00:00'
        >>> ConsoleProgressMonitorPlayer.format_interval(3600000)
        '1000:00:00'
        """
        mins, s = divmod(int(t), 60)
        h, m = divmod(mins, 60)
        if h:
            return '{0:d}:{1:02d}:{2:02d}'.format(h, m, s)
        else:
            return '{0:02d}:{1:02d}'.format(m, s)

    @property
    def info_header(self):
        return "{:>10}| {:>10}"

    @property
    def headers(self):
        return ["Iter", "Total-I"]

    def __call__(self, n, **kwargs):
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

        self.n = n

        res_str = self.output_format(*arguments)

        if not self.silent:
            flush_print(res_str, file=self.stream)

        self._total = n if self._total is None else max(n, self._total)

        return res_str

    def output_format(self, *arguments):
        return self.output_formatter.format(
            self.n, self.total, *arguments, self.progress
        )

    def start(self, *args, **kwargs):
        self.timer.start()

        res_str = self.index_prefix

        if not self.silent:
            print(res_str, file=self.stream)

        return res_str

    def end(self, n=None):
        if n is not None:
            self.total = n
        elif self._total is not None:
            self.total = self._total
        self(self.total)
        self.timer.end()
        if not self.silent:
            print("", file=self.stream)
        return ""

    @contextmanager
    def watching(self, *args, **kwargs):
        self.start(*args, **kwargs)
        yield
        self.end()

    @property
    def time(self):
        return self.timer.wall_time


class EBCPMP(ConsoleProgressMonitorPlayer):
    """
    Epoch-Batch Console Progress Monitor Player
    """

    def __init__(self, indexes: (dict, OrderedDict), values: (dict, None) = None,
                 total=NAN, silent=False, timer=None, stream=sys.stdout, total_epoch=NAN, *args, **kwargs):
        self.epoch = None
        self.total_epoch = total_epoch
        super(EBCPMP, self).__init__(
            indexes=indexes, values=values, total=total, silent=silent, timer=timer, stream=stream,
            *args, **kwargs
        )

    @property
    def info_header(self):
        return "{:>5}| {:>7}" + " " * 5 + "{:>10}" + " " * 2 + "{:>10}" + " " * 5

    @property
    def headers(self):
        return ["Epoch", "Total-E", "Batch", "Total-B"]

    def start(self, epoch, *args, **kwargs):
        self.epoch = epoch
        super(EBCPMP, self).start()

    def output_format(self, *arguments):
        return self.output_formatter.format(
            self.epoch, self.total_epoch, self.n, self.total, *arguments, self.progress
        )


class EpisodeCPMP(ConsoleProgressMonitorPlayer):
    @property
    def headers(self):
        return ["Episode", "Total-E"]


PLAYER_TYPE = {
    None: ConsoleProgressMonitorPlayer,
    "default": ConsoleProgressMonitorPlayer,
    "simple": ConsoleProgressMonitorPlayer,
    "epoch": EBCPMP,
    "episode": EpisodeCPMP,
}


class ConsoleProgressMonitor(ProgressMonitor):
    def __init__(self, indexes: (dict, OrderedDict), values: (dict, None) = None,
                 total=NAN, silent=False, player_type='default', *args, **kwargs):
        """

        Parameters
        ----------
        indexes: dict of list of str

        values: dict of dict of value

        total
        silent
        player_type
        args
        kwargs
        """
        super(ConsoleProgressMonitor, self).__init__(
            player=PLAYER_TYPE[player_type](
                indexes=indexes,
                values=values,
                total=total,
                silent=silent,
                *args,
                **kwargs
            )
        )
        self.silent = silent

    def __call__(self, iterator, *args, **kwargs):
        if self.silent:
            return iterator
        else:
            try:
                total = len(iterator)
                self.player.total = total
            except TypeError:
                pass
            self.player.start(*args, **kwargs)
            return IterableMIcing(iterator, self.player, self.player.end)

    @property
    def iteration_time(self):
        return self.player.time


if __name__ == '__main__':
    # def iter_fn(num):
    #     for i in range(num):
    #         yield i
    #
    #
    # import time
    # from longling import print_time
    #
    # values = {
    #     "Loss": {"l2": 0}
    # }
    # cp = ConsoleProgressMonitor({"Loss": ["l2"]}, values=values, player_type="epoch", total_epoch=2)
    #
    # with print_time():
    #     for e in range(2):
    #         for i in cp(iter_fn(100), e + 1):
    #             values["Loss"]["l2"] = i
    #             time.sleep(0.05)
    #
    #         print(cp.iteration_time)
    _values = {
        "Loss": {"l2": 0}
    }
    player = ConsoleProgressMonitorPlayer({"Loss": ["l2"]}, values=_values)
    with player.watching():
        for i in range(100):
            player(i, Loss={"l2": 1})
    print(player.time)
