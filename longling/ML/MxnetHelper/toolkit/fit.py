# coding: utf-8
# create by tongshiwei on 2020-11-12
import mxnet as mx
from mxnet import autograd

from longling.ML.MxnetHelper.toolkit.ctx import split_and_load

__all__ = ["fit_wrapper"]


def fit_wrapper(_fit_f):
    def fit_f(net, batch_size, batch_data,
              trainer, bp_loss_f, loss_function, loss_monitor=None,
              ctx=mx.cpu()):
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
        ctx_data = split_and_load(
            ctx, *batch_data,
            even_split=False
        )

        with autograd.record():
            for _data in ctx_data:
                bp_loss = _fit_f(
                    net, _data, bp_loss_f, loss_function, loss_monitor
                )
                assert bp_loss is not None
                bp_loss.backward()
        # todo: confirm whether the train step is equal to batch_size
        trainer.step(batch_size)

    return fit_f
