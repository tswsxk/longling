# coding: utf-8
# 2021/3/15 @ tongshiwei

def get_total_update_steps(update_epoch, batches_per_epoch, epoch_update_freq=1):
    return int(update_epoch * batches_per_epoch * epoch_update_freq)


def get_factor_lr_params(base_lr, update_epoch, batches_per_epoch, epoch_update_freq=1, stop_lr=None, discount=None):
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
        return stop_lr, discount
    elif discount:
        return base_lr * discount, discount
