# coding: utf-8
# 2021/3/15 @ tongshiwei

def get_total_update_steps(update_epoch, batches_per_epoch, epoch_update_freq=1):
    """

    Parameters
    ----------
    update_epoch
    batches_per_epoch
    epoch_update_freq

    Returns
    -------

    Examples
    --------
    >>> get_total_update_steps(10, 30)
    300
    >>> get_total_update_steps(10, 30, 2)
    600
    """
    return int(update_epoch * batches_per_epoch * epoch_update_freq)


def get_factor_lr_params(base_lr, update_epoch, batches_per_epoch, epoch_update_freq=1, stop_lr=None, discount=None):
    """

    Parameters
    ----------
    base_lr
    update_epoch
    batches_per_epoch
    epoch_update_freq
    stop_lr
    discount

    Returns
    -------

    Examples
    --------
    >>> get_factor_lr_params(0.01, 10, 30, 2, 0.001)
    (15, 0.9440608762859234, 0.001)
    >>> get_factor_lr_params(0.01, 10, 30, 2, discount=0.1)
    (15, 0.9440608762859234, 0.001)
    """
    assert epoch_update_freq

    total_update_steps = get_total_update_steps(
        update_epoch,
        batches_per_epoch,
        epoch_update_freq
    )
    step = batches_per_epoch // epoch_update_freq
    stop_lr, discount = _stop_lr(base_lr, stop_lr, discount)
    return step, pow(discount, step / total_update_steps), stop_lr


def _stop_lr(base_lr, stop_lr=None, discount=None):
    assert stop_lr or discount
    if stop_lr:
        return stop_lr, stop_lr / base_lr
    elif discount:
        return base_lr * discount, discount
