# coding: utf-8
# 2021/3/15 @ tongshiwei

def get_max_update(update_epoch, batches_per_epoch, warmup_steps=0):
    """

    Parameters
    ----------
    update_epoch: int
    batches_per_epoch: int
    warmup_steps: int

    Returns
    -------
    max_update: int
        maximum number of updates before the decay reaches final learning rate, which includs warmup steps
    Examples
    --------
    >>> get_max_update(10, 30)
    300
    >>> get_max_update(10, 30, 300)
    600
    """
    return update_epoch * batches_per_epoch + warmup_steps


def get_lr_factor(base_lr, step, max_update, final_lr=None, discount=None):
    """

    Parameters
    ----------
    base_lr
    step: int
        Changes the learning rate for every n updates.
    max_update: int
        maximum number of updates before the decay reaches final learning rate, which includs warmup steps
    final_lr:
        final learning rate after all steps

    discount

    Returns
    -------
    step: int
        Changes the learning rate for every n updates.
    factor:
    final_lr:
        final learning rate after all steps

    Examples
    --------
    >>> get_lr_factor(1, 10, 100, final_lr=1e-10)
    (10, 0.09999999999999999, 1e-10)
    >>> get_lr_factor(0.1, 10, 100, discount=1e-10)
    (10, 0.09999999999999999, 1.0000000000000001e-11)
    """
    final_lr, discount = get_final_lr(base_lr, final_lr, discount)
    return step, pow(discount, step / max_update), final_lr


def get_step(batches_per_epoch, epoch_update_freq=1):
    """
    Parameters
    ---------
    batches_per_epoch

    epoch_update_freq: int
        the frequence to change the learning rate (freq = 1 / (n times at one epoch) ).
        if epoch_update_freq <= 0, update at each step (batch)

    Returns
    -------
    step: int
        Changes the learning rate for every n updates.

    Examples
    -------
    >>> get_step(100, 1)
    100
    >>> get_step(100, 2)
    50
    """
    return batches_per_epoch // epoch_update_freq if epoch_update_freq >= 1 else 1


def get_final_lr(base_lr, final_lr=None, discount=None):
    """

    Parameters
    ----------
    base_lr
    final_lr:
        final learning rate after all steps
    discount

    Returns
    -------
    final_lr: int or float
        final learning rate after all steps
    discount:

    Examples
    --------
    >>> final_lr, discount = get_final_lr(0.1, 0.01)
    >>> round(final_lr, 2)
    0.01
    >>> round(discount, 2)
    0.1
    >>> final_lr, discount = get_final_lr(0.1, discount=0.1)
    >>> round(final_lr, 2)
    0.01
    >>> round(discount, 2)
    0.1
    """
    assert final_lr or discount
    if final_lr:
        return final_lr, final_lr / base_lr
    elif discount:
        return base_lr * discount, discount
