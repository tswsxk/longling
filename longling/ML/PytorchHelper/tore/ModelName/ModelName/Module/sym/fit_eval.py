# coding: utf-8
# create by tongshiwei on 2019-9-1
__all__ = ["fit_f", "eval_f"]

import torch
from tqdm import tqdm

from longling.ML.PytorchHelper import tensor2list


def _fit_f(_net, _data, bp_loss_f, loss_function, loss_monitor):
    data, label = _data
    output, _ = _net(data)
    bp_loss = None
    for name, func in loss_function.items():
        # todo modify the input to func
        loss = func(output, label)
        if name in bp_loss_f:
            bp_loss = loss
        loss_value = torch.mean(loss).item()
        if loss_monitor:
            loss_monitor.update(name, loss_value)
    return bp_loss


def eval_f(_net, test_data):
    ground_truth = []
    prediction = []

    def evaluation_function(y_true, y_pred):
        return 0

    for (data, data_mask, label, pick_index, label_mask) in tqdm(test_data, "evaluating"):
        with torch.no_grad():
            output, _ = _net(data, data_mask)
            pred = tensor2list(output)
            label = tensor2list(label)
            ground_truth.extend(label)
            prediction.extend(pred)
    return {
        "evaluation_name": evaluation_function(ground_truth, prediction)
    }


# #####################################################################
# ###     The following codes usually do not need modification      ###
# #####################################################################

def fit_f(net, batch_data,
          trainer, bp_loss_f, loss_function, loss_monitor=None
          ):
    """
    Defined how each step of batch train goes

    Parameters
    ----------
    net: HybridBlock
        The network which has been initialized
        or loaded from the existed model
    batch_data: Iterable
        The batch data for train
    trainer:
        The trainer used to update the parameters of the net
    bp_loss_f: dict with only one value and one key
        The function to compute the loss for the procession
        of back propagation
    loss_function: dict of function
        Some other measurement in addition to bp_loss_f
    loss_monitor: LossMonitor
        Default to ``None``
    ctx: Context or list of Context
        Defaults to ``mx.cpu()``.

    Returns
    -------

    """
    bp_loss = _fit_f(
        net, batch_data, bp_loss_f, loss_function, loss_monitor
    )
    assert bp_loss is not None
    trainer.zero_grad()
    bp_loss.backward()
    trainer.step()
