# coding: utf-8
# 2021/3/13 @ tongshiwei


__all__ = ["fit_wrapper"]


def fit_wrapper(_fit_f):
    def fit_f(net, batch_data, loss_function, trainer,
              batch_size=None, loss_monitor=None, *args, **kwargs):
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
            The function to compute the loss for the procession
            of back propagation
        loss_monitor: LossMonitor
            Default to ``None``

        Returns
        -------

        """
        batch_size = batch_size if batch_size is not None else len(batch_data[0])

        bp_loss = _fit_f(net, batch_data, loss_function, loss_monitor, batch_size)

        trainer.zero_grad()
        bp_loss.backward()
        trainer.step()

    return fit_f
