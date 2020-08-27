# coding: utf-8
# 2020/4/13 @ tongshiwei

# incubating

import logging
import json
from longling.lib.stream import as_out_io, PATH_IO_TYPE
from collections import OrderedDict
from longling.lib.formatter import table_format, series_format
from longling.lib.candylib import as_list

__all__ = ["eval_format", "EvalFMT", "EpochEvalFMT", "EpisodeEvalFMT", "result_format"]


def _to_dict(name_value: (dict, tuple)) -> dict:
    """Make sure the name_value a dict object"""
    return dict(name_value) if isinstance(name_value, tuple) else name_value


class EvalFMT(object):
    def __init__(self, logger=logging.getLogger(), dump_file: (PATH_IO_TYPE, None) = False,
                 col: (int, None) = None, **kwargs):
        self.logger = logger
        if dump_file is not False:
            # clean file
            with as_out_io(dump_file):
                pass
        self.log_f = dump_file
        self.col = col

    @classmethod
    def _loss_format(cls, name, value):
        return "%s: %s" % (name, value)

    def loss_format(self, loss_name_value):
        msg = "Loss - "
        for name, value in loss_name_value.items():
            msg += self._loss_format(name, value)
        return msg, loss_name_value

    @classmethod
    def format(cls, tips: str = None,
               iteration: int = None, train_time: float = None, loss_name_value: dict = None,
               eval_name_value: dict = None,
               extra_info: (dict, tuple) = None, keep: (set, str) = "msg",
               logger=logging.getLogger(), dump_file: (PATH_IO_TYPE, None) = False,
               col: (int, None) = None,
               *args, **kwargs):
        return cls(logger=logger, dump_file=dump_file, col=col)(
            tips=tips, iteration=iteration, train_time=train_time, loss_name_value=loss_name_value,
            eval_name_value=eval_name_value, extra_info=extra_info,
            dump=dump_file is not False, keep=keep, *args, **kwargs
        )

    @property
    def iteration_name(self):
        return "Iteration"

    @property
    def iteration_fmt(self):
        return self.iteration_name + " [{:d}]"

    def __call__(self, tips: str = None,
                 iteration: int = None, train_time: float = None, loss_name_value: dict = None,
                 eval_name_value: dict = None,
                 extra_info: (dict, tuple) = None,
                 dump: bool = True, keep: (set, str) = "data", *args, **kwargs):
        msg = []
        data = {}

        if tips is not None:
            msg.append("%s" % tips)

        if iteration is not None:
            msg.append(self.iteration_fmt.format(iteration))
            data[self.iteration_name] = iteration

        if train_time is not None:
            msg.append("Train Time-%.3fs" % train_time)
            data['train_time'] = train_time

        if loss_name_value is not None:
            loss_name_value = _to_dict(loss_name_value)
            assert isinstance(
                loss_name_value, dict
            ), "loss_name_value should be None, dict or tuple, " \
               "now is %s" % type(loss_name_value)
            _msg, _data = self.loss_format(loss_name_value)

            msg.append(
                _msg
            )
            data.update(
                _data
            )

        if extra_info is not None:
            extra_info = _to_dict(extra_info)
            assert isinstance(
                extra_info, dict
            ), "extra_info should be None, dict or tuple, " \
               "now is %s" % type(extra_info)
            msg.append(str(extra_info))
            data.update(extra_info)

        msg = ["\t".join([m for m in msg if m])]

        if eval_name_value is not None:
            eval_name_value = _to_dict(eval_name_value)
            assert isinstance(
                eval_name_value, dict
            ), "eval_name_value should be None, dict or tuple, " \
               "now is %s" % type(eval_name_value)
            msg.append(
                result_format(eval_name_value, col=self.col)
            )
            data.update(
                eval_name_value
            )

        msg = "\n".join([m for m in msg if m])

        if dump:
            logger = kwargs.get('logger', self.logger)
            logger.info("\n" + msg)
            log_f = kwargs.get('log_f', self.log_f)
            if log_f is not False:
                try:
                    with as_out_io(log_f, "a") as wf:
                        print(json.dumps(data, ensure_ascii=False), file=wf)
                except Exception as e:  # pragma: no cover
                    logger.warning(e)

        if keep is None:
            return msg
        elif isinstance(keep, str):
            keep = set(as_list(keep))

        if "msg" in keep and "data" in keep:
            return msg, data
        elif "msg" in keep:
            return msg
        elif "data" in keep:
            return data


def result_format(data: dict, col=None):
    table = OrderedDict()
    series = OrderedDict()
    for key, value in data.items():
        if isinstance(value, dict):
            table[key] = value
        else:
            series[key] = value

    _ret = []
    if table:
        _ret.append(table_format(table))
    if series:
        _ret.append(series_format(series, col=col))

    return "\n".join(_ret)


eval_format = EvalFMT.format


class EpochEvalFMT(EvalFMT):
    @property
    def iteration_name(self):
        return "Epoch"


class EpisodeEvalFMT(EvalFMT):
    @property
    def iteration_name(self):
        return "Episode"


if __name__ == '__main__':
    import numpy as np
    import logging
    from longling.ML.metrics import classification_report

    logging.getLogger().setLevel(logging.INFO)

    y_true = np.array([0, 0, 1, 1, 2, 1])
    y_pred = np.array([2, 1, 0, 1, 1, 0])
    y_score = np.array([
        [0.15, 0.4, 0.45],
        [0.1, 0.9, 0.0],
        [0.33333, 0.333333, 0.333333],
        [0.15, 0.4, 0.45],
        [0.1, 0.9, 0.0],
        [0.33333, 0.333333, 0.333333]
    ])

    # print(result_format(classification_report(y_true, y_pred, y_score)))
    EvalFMT.format(
        iteration=30,
        eval_name_value=classification_report(y_true, y_pred, y_score)
    )
