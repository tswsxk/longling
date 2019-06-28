# coding: utf-8
# created by tongshiwei on 18-2-5

import codecs
import json
import logging
import re

from longling.base import string_types
from longling.lib.stream import wf_open

__all__ = ["EvalFormatter", "MultiClassEvalFormatter"]


def _to_dict(name_value):
    return dict(name_value) if isinstance(name_value, tuple) else name_value


class EvalFormatter(object):
    def __init__(self, logger=logging.getLogger(), dump_file=None, **kwargs):
        self.logger = logger
        if dump_file is not None and isinstance(dump_file, string_types):
            # clean file
            wf_open(dump_file, **kwargs).close()
        self.log_f = dump_file

    @staticmethod
    def _loss_format(name, value):
        return "%s: %s" % (name, value)

    def loss_format(self, loss_name_value):
        msg = "Loss - "
        for name, value in loss_name_value.items():
            msg += self._loss_format(name, value)
        return msg, loss_name_value

    @staticmethod
    def _eval_format(name, value):
        return "Evaluation %s: %s" % (name, value)

    def eval_format(self, eval_name_value):
        msg = []
        for name, value in eval_name_value.items():
            msg.append(self._eval_format(name, value))
        msg = "\t".join([m for m in msg if m])
        data = eval_name_value
        return msg, data

    def __call__(self, tips=None,
                 epoch=None, train_time=None, loss_name_value=None,
                 eval_name_value=None,
                 extra_info=None,
                 dump=True,
                 *args, **kwargs
                 ):

        msg = []
        data = {}

        if tips is not None:
            msg.append("%s" % tips)

        if epoch is not None:
            msg.append("Epoch [%d]:" % epoch)
            data['Epoch'] = epoch

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
            _msg, _data = self.eval_format(eval_name_value)
            msg.append(
                _msg
            )
            data.update(
                _data
            )

        msg = "\n".join([m for m in msg if m])

        if dump:
            logger = kwargs.get('logger', self.logger)
            logger.info(msg)
            log_f = kwargs.get('log_f', self.log_f)
            if log_f is not None:
                try:
                    if log_f is not None and isinstance(log_f, string_types):
                        log_f = codecs.open(log_f, "a", encoding="utf-8")
                        print(json.dumps(data, ensure_ascii=False), file=log_f)
                        log_f.close()
                    else:
                        print(json.dumps(data, ensure_ascii=False), file=log_f)
                except Exception as e:
                    logger.warning(e)
        return msg, data


class MultiClassEvalFormatter(EvalFormatter):
    def eval_format(self, eval_name_value):
        msg = []
        data = {}

        multi_class_pattern = re.compile(r".+_\d+")

        prf = {}
        eval_ids = set()

        for name, value in sorted(eval_name_value.items()):
            if multi_class_pattern.match(name) is not None:
                try:
                    eval_id, class_id = name.split("_")
                except ValueError:
                    continue
                if class_id not in prf:
                    prf[class_id] = {}
                prf[class_id][eval_id] = value
                eval_ids.add(eval_id)
            else:
                msg.append(self._eval_format(name, value))
                data[name] = value

        msg = "\t".join([m for m in msg if m])
        if msg:
            msg += '\n'
        if prf:
            avg = {eval_id: [] for eval_id in eval_ids}
            for class_id in [str(k) for k in
                             sorted([int(k) for k in prf.keys()])]:
                for eval_id, values in avg.items():
                    values.append(prf[class_id][eval_id])
                msg += "--- Category %s" % class_id
                msg_res = sorted(prf[class_id].items(), reverse=True)
                msg += ("\t{}={:.10f}" * len(prf[class_id])).format(
                    *sum(msg_res, ())) + "\n"
            avg = {
                eval_id: sum(values) / len(values)
                for eval_id, values in avg.items()
            }
            msg += "--- Category_Avg "
            msg_res = sorted(avg.items(), reverse=True)
            msg += ("\t{}={:.10f}" * len(avg)).format(*sum(msg_res, ()))
            prf['avg'] = avg
            data['prf'] = prf

        return msg, data


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    formatter = MultiClassEvalFormatter()
    print(formatter(
        eval_name_value={
            "Acuuracy": 0.5,
            "precision_1": 10, "precision_0": 20,
            "recall_0": 1, "recall_1": 2
        }
    )[0])
