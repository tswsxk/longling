# coding: utf-8
# 2020/4/13 @ tongshiwei

# incubating

import logging
import json
from longling.lib.stream import wf_open, as_out_io
from collections import OrderedDict
from longling.lib.formatter import table_format, series_format
from longling.lib.candylib import as_list


def _to_dict(name_value: (dict, tuple)) -> dict:
    """Make sure the name_value a dict object"""
    return dict(name_value) if isinstance(name_value, tuple) else name_value


class EvalFMT(object):
    def __init__(self, logger=logging.getLogger(), dump_file: (str, None) = False,
                 col: (int, None) = None, **kwargs):
        self.logger = logger
        if dump_file is not None and isinstance(dump_file, str):
            # clean file
            wf_open(dump_file, **kwargs).close()
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
    def format(cls, tips=None,
               iteration=None, train_time=None, loss_name_value=None,
               eval_name_value: dict = None,
               extra_info=None,
               dump=True, logger=logging.getLogger(), dump_file: (str, None) = False,
               col: (int, None) = None,
               *args, **kwargs):
        return cls(logger=logger, dump_file=dump_file, col=col)(
            tips=tips, iteration=iteration, train_time=train_time, loss_name_value=loss_name_value,
            eval_name_value=eval_name_value, dump=dump
        )

    @property
    def iteration_name(self):
        return "Iteration"

    @property
    def iteration_fmt(self):
        return self.iteration_name + " [{:d}]"

    def __call__(self, tips=None,
                 iteration=None, train_time=None, loss_name_value=None,
                 eval_name_value: dict = None,
                 extra_info=None,
                 dump=True, keep: (set, str) = None, *args, **kwargs):
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
            msg.append(extra_info.items())
            data.update(extra_info)

        msg = ["\t".join([m for m in msg if m])]

        if eval_name_value is not None:
            eval_name_value = _to_dict(eval_name_value)
            assert isinstance(
                eval_name_value, dict
            ), "eval_name_value should be None, dict or tuple, " \
               "now is %s" % type(eval_name_value)
            msg.append(
                result_format(eval_name_value)
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
                except Exception as e:
                    logger.warning(e)

        if keep is None:
            return msg
        elif isinstance(keep, str):
            keep = set(as_list(keep))

        if "msg" in keep:
            return msg
        if "data" in keep:
            return data


def result_format(data: dict):
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
        _ret.append(series_format(series))

    return "\n".join(_ret)


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
