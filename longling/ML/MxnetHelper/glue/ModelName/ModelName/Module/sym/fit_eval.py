# coding: utf-8
# create by tongshiwei on 2019-9-1
__all__ = ["fit_f", "eval_f"]

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd
from tqdm import tqdm

from longling.ML.MxnetHelper.toolkit.ctx import split_and_load


def _fit_f(_net, _data, loss_function, loss_monitor=None):
    data, label = _data
    # todo modify the input to net
    output = _net(data)
    loss_list = []
    for name, func in loss_function.items():
        loss = func(output, label)
        loss_list.append(loss)
        # loss_value = nd.mean(loss).asscalar()
        # if loss_monitor:
        #     loss_monitor.update(name, loss_value)
    return sum(loss_list)


def eval_f(_net, test_data, ctx=mx.cpu()):
    ground_truth = []
    prediction = []

    def evaluation_function(y_true, y_pred):
        return {"evaluation_name": 0}

    for batch_data in tqdm(test_data, "evaluating"):
        ctx_data = split_and_load(
            ctx, *batch_data,
            even_split=False
        )
        for (data, label) in ctx_data:
            output = _net(data)
            pred = output
            ground_truth.extend(label.asnumpy().tolist())
            prediction.extend(pred.asnumpy().tolist())

    return evaluation_function(ground_truth, prediction)


def fit_f(net, batch_size, batch_data,
          trainer, loss_function, loss_monitor=None,
          ctx=mx.cpu(), fit_step_func=_fit_f):
    """
    Defined how each step of batch train goes

    Parameters
    ----------
    net: HybridBlock
        The network which has been initialized
        or loaded from the existed model
    batch_size: int
            The size of each batch
    batch_data: Iterable
        The batch data for train
    trainer:
        The trainer used to update the parameters of the net
    loss_function: dict of function
        The functions to compute the loss for the procession
        of back propagation
    loss_monitor: LossMonitor
        Default to ``None``
    ctx: Context or list of Context
        Defaults to ``mx.cpu()``.
    fit_step_func: function

    Returns
    -------

    """
    ctx_data = split_and_load(
        ctx, *batch_data,
        even_split=False
    )

    with autograd.record():
        for _data in ctx_data:
            bp_loss = fit_step_func(
                net, _data, loss_function, loss_monitor
            )
            bp_loss.backward()

    # todo: confirm whether the train step is equal to batch_size
    if batch_size is not None:
        trainer.step(batch_size)
    else:
        trainer.step(len(batch_data[0]))
