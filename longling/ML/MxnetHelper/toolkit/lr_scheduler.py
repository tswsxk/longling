# coding: utf-8
# 2020/8/18 @ tongshiwei

import logging
import numpy as np
import scipy.stats as st
from mxnet.lr_scheduler import LRScheduler, FactorScheduler, PolyScheduler, MultiFactorScheduler, CosineScheduler

from longling.ML.DL import get_max_update, get_lr_factor, get_final_lr, get_step

__all__ = ["get_lr_scheduler", "plot_schedule", "get_max_update"]


class _LRScheduler(LRScheduler):
    def __repr__(self):
        return str(self.state_dict)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def state_dict(self):
        raise NotImplementedError

    @classmethod
    def init(cls, base_lr, batches_per_epoch=None, update_epoch=None, epoch_update_freq=1, warmup_epoch=0,
             discount=None, *args, **kwargs):

        if warmup_epoch:
            kwargs.update({"warmup_steps": warmup_epoch * batches_per_epoch})

        if discount is not None:
            final_lr, discount = get_final_lr(base_lr, discount=discount)
            kwargs.update({"final_lr": final_lr})

        return kwargs


class _FactorScheduler(FactorScheduler, _LRScheduler):
    """update at every *n* steps (batches)"""

    def __repr__(self):
        return str(self.state_dict)

    @property
    def state_dict(self):
        return {
            "scheduler": "FactorScheduler",
            "base_lr": self.base_lr,
            "step": self.step,
            "stop_factor_lr": self.stop_factor_lr,
            "warmup_mode": self.warmup_mode,
            "warmup_begin_lr": self.warmup_begin_lr,
            "warmup_steps": self.warmup_steps,
        }

    @classmethod
    def init(cls, base_lr, batches_per_epoch=None, update_epoch=None, epoch_update_freq=1, warmup_epoch=0,
             discount=None, **kwargs):
        kwargs.update(super(_FactorScheduler, cls).init(
            base_lr,
            warmup_epoch=warmup_epoch,
            batches_per_epoch=batches_per_epoch,
            discount=discount
        ))

        if batches_per_epoch is not None and update_epoch is not None:
            step, factor, stop_factor_lr = get_lr_factor(
                base_lr,
                step=get_step(batches_per_epoch, epoch_update_freq),
                max_update=get_max_update(update_epoch, batches_per_epoch, kwargs.get("warmup_steps", 0)),
                final_lr=kwargs.get("final_lr", kwargs.get("stop_factor_lr", None)),
                discount=discount
            )
            kwargs.update({"step": step, "factor": factor, "stop_factor_lr": stop_factor_lr})

        elif "max_update" in kwargs:
            max_update = kwargs.pop("max_update")
            step, factor, stop_factor_lr = get_lr_factor(
                base_lr,
                step=kwargs["step"],
                max_update=max_update,
                final_lr=kwargs.get("final_lr", kwargs.get("stop_factor_lr", None)),
                discount=discount
            )

            kwargs.update({"step": step, "factor": factor, "stop_factor_lr": stop_factor_lr})

        if "final_lr" in kwargs:
            kwargs.pop("final_lr")

        return cls(base_lr=base_lr, **kwargs)


class _PolyScheduler(PolyScheduler, _LRScheduler):
    """update at each step (batch)"""

    @property
    def state_dict(self):
        return {
            "scheduler": "PolyScheduler",
            "base_lr": self.base_lr,
            "step": 1,
            "max_steps": self.max_steps,
            "final_lr": self.final_lr,
            "warmup_mode": self.warmup_mode,
            "warmup_begin_lr": self.warmup_begin_lr,
            "warmup_steps": self.warmup_steps,
        }

    @classmethod
    def init(cls, base_lr, batches_per_epoch=None, update_epoch=None, epoch_update_freq=0, warmup_epoch=0,
             discount=None, *args, **kwargs):
        kwargs = super(_PolyScheduler, cls).init(
            base_lr,
            warmup_epoch=warmup_epoch,
            batches_per_epoch=batches_per_epoch,
            discount=discount
        )

        if batches_per_epoch is not None and update_epoch is not None:
            max_update = get_max_update(update_epoch, batches_per_epoch, kwargs.get("warmup_steps", 0))
            kwargs.update({"max_update": max_update})

        return cls(base_lr=base_lr, **kwargs)


class _CosineScheduler(CosineScheduler, _LRScheduler):
    """update at each step (batch)"""

    @property
    def state_dict(self):
        return {
            "scheduler": "CosineScheduler",
            "base_lr": self.base_lr,
            "step": 1,
            "max_steps": self.max_steps,
            "final_lr": self.final_lr,
            "warmup_mode": self.warmup_mode,
            "warmup_begin_lr": self.warmup_begin_lr,
            "warmup_steps": self.warmup_steps,
        }

    @classmethod
    def init(cls, base_lr, batches_per_epoch=None, update_epoch=None, epoch_update_freq=0, warmup_epoch=0,
             discount=None, *args, **kwargs):
        batches_per_epoch = epoch_update_freq if batches_per_epoch is None else batches_per_epoch

        kwargs.update(super(_CosineScheduler, cls).init(
            base_lr,
            warmup_epoch=warmup_epoch,
            batches_per_epoch=batches_per_epoch,
            discount=discount
        ))

        if batches_per_epoch is not None and update_epoch is not None:
            max_update = get_max_update(batches_per_epoch, update_epoch, kwargs.get("warmup_steps", 0))
            kwargs.update({"max_update": max_update})

        return cls(base_lr=base_lr, **kwargs)


class _MultiFactorScheduler(MultiFactorScheduler, _LRScheduler):
    """Reduce the learning rate by given a list of steps"""

    @property
    def state_dict(self):
        return {
            "scheduler": "MultiFactorScheduler",
            "base_lr": self.base_lr,
            "steps": self.step,
            "warmup_mode": self.warmup_mode,
            "warmup_begin_lr": self.warmup_begin_lr,
            "warmup_steps": self.warmup_steps,
        }

    @classmethod
    def init(cls, base_lr, batches_per_epoch=None, update_epoch=None, epoch_update_freq=1, warmup_epoch=0,
             discount=None, *args, **kwargs):

        kwargs.update(super(_MultiFactorScheduler, cls).init(
            base_lr,
            warmup_epoch=warmup_epoch,
            batches_per_epoch=batches_per_epoch,
            discount=discount
        ))

        if batches_per_epoch is not None and update_epoch is not None:
            assert isinstance(update_epoch, list)

            if kwargs.get("acc_step", False):
                kwargs.pop("acc_step")
                acc_update_epoch = [update_epoch[0]]
                for i, _step in enumerate(update_epoch[1:]):
                    acc_update_epoch.append(acc_update_epoch[i] + _step)
                update_epoch = acc_update_epoch

            _, factor, stop_factor_lr = get_lr_factor(
                base_lr,
                step=len(update_epoch) - 1,
                max_update=update_epoch[-2],
                final_lr=kwargs.get("final_lr", kwargs.get("stop_factor_lr", None)),
                discount=discount
            )
            kwargs.update({"factor": factor})

            step = [_update_epoch * batches_per_epoch for _update_epoch in update_epoch[:-1]]
            kwargs.update({"step": step})

        elif kwargs.get("acc_step", False):
            kwargs.pop("acc_step")
            step = kwargs.pop("step")
            acc_step = [step[0]]
            for i, _step in enumerate(step[1:]):
                acc_step.append(acc_step[i] + _step)
            kwargs.update({"step": acc_step})

        if "final_lr" in kwargs:
            kwargs.pop("final_lr")
        return cls(base_lr=base_lr, **kwargs)


class LinearScheduler(LRScheduler):
    """update at every *n* steps (batches)"""

    def __init__(self, max_update, step=1, base_lr=0.01, final_lr=0,
                 warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):
        super(LinearScheduler, self).__init__(
            base_lr, warmup_steps=warmup_steps, warmup_begin_lr=warmup_begin_lr, warmup_mode=warmup_mode
        )
        assert isinstance(max_update, int)
        if max_update < 1:
            raise ValueError("maximum number of updates must be strictly positive")
        self.max_update = max_update
        self.final_lr = final_lr
        self.count = 0
        self.step = step
        self.max_steps = self.max_update // self.step
        self.increase = (self.final_lr - self.base_lr) / ((self.max_steps - 1) if self.max_steps > 1 else 1)

    def __call__(self, num_update):
        if num_update < self.warmup_steps:
            return self.get_warmup_lr(num_update)
        while num_update > self.count + self.step:
            self.count += self.step
            self.base_lr += self.increase
        if self.base_lr < self.final_lr:
            return self.final_lr
        return self.base_lr


class _LinearScheduler(LinearScheduler, _LRScheduler):
    @property
    def state_dict(self):
        return {
            "scheduler": "LinearScheduler",
            "base_lr": self.base_lr,
            "max_steps": self.max_steps,
            "warmup_mode": self.warmup_mode,
            "warmup_begin_lr": self.warmup_begin_lr,
            "warmup_steps": self.warmup_steps,
        }

    @classmethod
    def init(cls, base_lr, batches_per_epoch=None, update_epoch=None, epoch_update_freq=1, warmup_epoch=0,
             discount=None, *args, **kwargs):
        batches_per_epoch = epoch_update_freq if batches_per_epoch is None else batches_per_epoch

        kwargs.update(super(_LinearScheduler, cls).init(
            base_lr,
            warmup_epoch=warmup_epoch,
            batches_per_epoch=batches_per_epoch,
            discount=discount
        ))
        if batches_per_epoch is not None and update_epoch is not None:
            kwargs.update({"step": get_step(batches_per_epoch, epoch_update_freq)})
            kwargs.update(
                {"max_update": get_max_update(update_epoch, batches_per_epoch, kwargs.get("warmup_steps", 0))}
            )
        return cls(base_lr=base_lr, **kwargs)


class NormScheduler(LRScheduler):
    """update at every *n* steps (batches)"""

    def __init__(self,
                 max_update, step=1,
                 base_lr=0.01, final_lr=1e-8,
                 lr_loc=None, lr_scale=None,
                 min_lr=None, max_lr=None,
                 epsilon=1e-10,
                 warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):

        if lr_loc is not None:
            self.lr_loc = lr_loc
            self.lr_scale = lr_scale if lr_scale is not None else 2 * lr_loc
            base_lr = self.lr_loc + 2 * self.lr_scale
        else:
            self.lr_loc = base_lr / 2 if lr_loc is None else lr_loc
            self.lr_scale = lr_scale if lr_scale is not None else (base_lr - self.lr_loc) / 2
        self.min_lr = max(final_lr, min_lr if min_lr is not None else (-2 * self.lr_scale + self.lr_loc))
        self.max_lr = min(max_lr if max_lr is not None else (2 * self.lr_scale + self.lr_loc), base_lr)
        self.dis = st.norm(loc=self.lr_loc, scale=self.lr_scale)
        self._p = 0
        self.epsilon = epsilon
        super(NormScheduler, self).__init__(
            base_lr=self.get_lr(), warmup_steps=warmup_steps,
            warmup_begin_lr=warmup_begin_lr, warmup_mode=warmup_mode,
        )
        self.step = step
        self.max_steps = max_update / self.step
        self.p_increase = 1 / (self.max_steps - 1 if self.max_steps > 1 else 1)
        self.count = 0

    @property
    def p(self):
        return np.clip(self.epsilon, self._p, 1 - self.epsilon)

    def get_lr(self):
        lr = self.dis.isf(self.p)
        if not self.min_lr <= lr <= self.max_lr:
            lr = np.clip(lr, self.min_lr, self.max_lr)
        return lr

    def __call__(self, num_update):
        if num_update < self.warmup_steps:
            return self.get_warmup_lr(num_update)

        # NOTE: use while rather than if  (for continuing training via load_epoch)
        while num_update > self.count + self.step:
            self.count += self.step
            self._p += self.p_increase
            self.base_lr = self.get_lr()
            logging.info("Update[%d]: Change learning rate to %0.5e",
                         num_update, self.base_lr)
        return self.base_lr


class _NormScheduler(NormScheduler, _LRScheduler):
    @property
    def state_dict(self):
        return {
            "scheduler": "NormScheduler",
            "lr_loc": self.lr_loc,
            "step": self.step,
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "lr_scale": self.lr_scale,
            "warmup_mode": self.warmup_mode,
            "warmup_begin_lr": self.warmup_begin_lr,
            "warmup_steps": self.warmup_steps,
        }

    @classmethod
    def init(cls, base_lr=None, batches_per_epoch=None, update_epoch=None, epoch_update_freq=1, warmup_epoch=0,
             discount=None, *args, **kwargs):
        batches_per_epoch = epoch_update_freq if batches_per_epoch is None else batches_per_epoch

        kwargs.update(super(_NormScheduler, cls).init(
            base_lr,
            warmup_epoch=warmup_epoch,
            batches_per_epoch=batches_per_epoch,
            discount=discount,
        ))
        if batches_per_epoch is not None and update_epoch is not None:
            max_update = int(batches_per_epoch * update_epoch) + kwargs.get("warmup_steps", 0)
            kwargs.update({"max_update": max_update, "step": get_step(batches_per_epoch, epoch_update_freq)})
        return cls(base_lr=base_lr, **kwargs)


SCHEDULERS = {
    "linear": _LinearScheduler,
    "factor": _FactorScheduler,
    "poly": _PolyScheduler,
    "multifactor": _MultiFactorScheduler,
    "cosine": _CosineScheduler,
    "norm": _NormScheduler,
}


def get_lr_scheduler(scheduler: (str, LRScheduler) = "cosine", logger=logging, update_params=None, adaptive=True,
                     **kwargs):
    """

    Parameters
    ----------
    scheduler
    logger
    update_params
    adaptive
    kwargs

    Other Parameters
    ----------------
    base_lr: int or float
    learning_rate:
        alias of base_lr
    lr:
        alias of base_lr
    batches_per_epoch: int

    update_epoch: int or list of int
        Changes the learning rate for total n epoch (equal or less than the training epoch).
        Specially, for MultiFactorScheduler, The list of update epochs to schedule a change,
        notice the last one element indicates the total training epoch, i.e., [c1, ..., cn, end_epoch]
    epoch_update_freq: int
        default to 1
        when set to 0, update_freq will be the same as batches_per_epoch (i.e., update once at an epoch)
    max_update: int
        maximum number of updates before the decay reaches final learning rate, which includs warmup steps
    total_step: int
        alias for max_update
    step: int or list of int
        Changes the learning rate for every n updates.
        Specially, for MultiFactorScheduler, The list of steps to schedule a change
    acc_step: bool
        Only for MultiFactorScheduler, whether calculate the accumulative steps based on step
    final_lr: int or float
        final learning rate after all steps
    warmup_epoch: int
    warmup_steps: int
        number of warmup steps used before this scheduler starts decay
    warmup_begin_lr: float
        if using warmup, the learning rate from which it starts warming up
    warmup_mode: str
        warmup can be done in two modes.
        'linear' mode gradually increases lr with each step in equal increments
        'constant' mode keeps lr at warmup_begin_lr for warmup_steps

    Returns
    -------
    scheduler: _LRScheduler or LRScheduler

    Examples
    --------
    >>> get_lr_scheduler(max_update=100, lr=0.01)  # doctest: +NORMALIZE_WHITESPACE
    {'scheduler': 'CosineScheduler', 'base_lr': 0.01, 'step': 1,
    'max_steps': 100, 'final_lr': 0, 'warmup_mode': 'linear', 'warmup_begin_lr': 0, 'warmup_steps': 0}
    >>> get_lr_scheduler("cosine", batches_per_epoch=10, update_epoch=10, lr=0.01)  # doctest: +NORMALIZE_WHITESPACE
    {'scheduler': 'CosineScheduler', 'base_lr': 0.01, 'step': 1,
    'max_steps': 100, 'final_lr': 0, 'warmup_mode': 'linear', 'warmup_begin_lr': 0, 'warmup_steps': 0}
    >>> get_lr_scheduler("CosineScheduler", max_update=100,
    ... base_lr=0.01, adaptive=False)  # doctest: +ELLIPSIS
    <mxnet.lr_scheduler.CosineScheduler...
    >>> get_lr_scheduler("norm", base_lr=1, discount=0.01, max_update=100)   # doctest: +NORMALIZE_WHITESPACE
    {'scheduler': 'NormScheduler', 'lr_loc': 0.5, 'step': 1, 'max_lr': 1.0, 'min_lr': 0.01, 'lr_scale': 0.25,
    'warmup_mode': 'linear', 'warmup_begin_lr': 0, 'warmup_steps': 0}
    >>> get_lr_scheduler("norm", base_lr=1, discount=0.01, lr_loc=0.02,
    ...     batches_per_epoch=100, update_epoch=10, epoch_update_freq=10,
    ... )  # doctest: +NORMALIZE_WHITESPACE
    {'scheduler': 'NormScheduler', 'lr_loc': 0.02, 'step': 10, 'max_lr': 0.1,
    'min_lr': 0.01, 'lr_scale': 0.04, 'warmup_mode': 'linear', 'warmup_begin_lr': 0, 'warmup_steps': 0}
    >>> get_lr_scheduler("linear", learning_rate=0.01, discount=0.1,
    ...     max_update=20, step=2)  # doctest: +NORMALIZE_WHITESPACE
    {'scheduler': 'LinearScheduler', 'base_lr': 0.01, 'max_steps': 10,
    'warmup_mode': 'linear', 'warmup_begin_lr': 0, 'warmup_steps': 0}
    >>> get_lr_scheduler("linear", learning_rate=0.01, discount=0.1,
    ...     update_epoch=2, warmup_epoch=1, batches_per_epoch=10)  # doctest: +NORMALIZE_WHITESPACE
    {'scheduler': 'LinearScheduler', 'base_lr': 0.01, 'max_steps': 3,
    'warmup_mode': 'linear', 'warmup_begin_lr': 0, 'warmup_steps': 10}
    >>> get_lr_scheduler("factor", learning_rate=0.01,
    ...     discount=0.1, max_update=10, step=2)  # doctest: +NORMALIZE_WHITESPACE
    {'scheduler': 'FactorScheduler', 'base_lr': 0.01, 'step': 2, 'stop_factor_lr': 0.001, 'warmup_mode': 'linear',
    'warmup_begin_lr': 0, 'warmup_steps': 0}
    >>> get_lr_scheduler("factor", learning_rate=0.01, discount=0.1,
    ...     update_epoch=2, warmup_epoch=1, batches_per_epoch=10)  # doctest: +NORMALIZE_WHITESPACE
    {'scheduler': 'FactorScheduler', 'base_lr': 0.01, 'step': 10, 'stop_factor_lr': 0.001,
    'warmup_mode': 'linear', 'warmup_begin_lr': 0, 'warmup_steps': 10}
    >>> get_lr_scheduler("poly", learning_rate=0.01, discount=0.1,
    ...     update_epoch=2, warmup_epoch=1, batches_per_epoch=10)  # doctest: +NORMALIZE_WHITESPACE
    {'scheduler': 'PolyScheduler', 'base_lr': 0.01, 'step': 1, 'max_steps': 20,
    'final_lr': 0.001, 'warmup_mode': 'linear', 'warmup_begin_lr': 0, 'warmup_steps': 10}
    >>> get_lr_scheduler("multifactor", learning_rate=0.01, discount=0.1,
    ...     step=[20, 30, 40])  # doctest: +NORMALIZE_WHITESPACE
    {'scheduler': 'MultiFactorScheduler', 'base_lr': 0.01, 'steps': [20, 30, 40],
    'warmup_mode': 'linear', 'warmup_begin_lr': 0, 'warmup_steps': 0}
    >>> get_lr_scheduler("multifactor", learning_rate=0.01, discount=0.1,
    ...     step=[20, 10, 10], acc_step=True)  # doctest: +NORMALIZE_WHITESPACE
    {'scheduler': 'MultiFactorScheduler', 'base_lr': 0.01, 'steps': [20, 30, 40],
    'warmup_mode': 'linear', 'warmup_begin_lr': 0, 'warmup_steps': 0}
    >>> get_lr_scheduler("multifactor", learning_rate=0.01, discount=0.1,
    ...     update_epoch=[2, 3, 4], warmup_epoch=1, batches_per_epoch=10)  # doctest: +NORMALIZE_WHITESPACE
    {'scheduler': 'MultiFactorScheduler', 'base_lr': 0.01, 'steps': [20, 30],
    'warmup_mode': 'linear', 'warmup_begin_lr': 0, 'warmup_steps': 10}
    >>> get_lr_scheduler("multifactor", learning_rate=0.01, discount=0.1, update_epoch=[2, 1, 1],
    ...     warmup_epoch=1, acc_step=True, batches_per_epoch=10)  # doctest: +NORMALIZE_WHITESPACE
    {'scheduler': 'MultiFactorScheduler', 'base_lr': 0.01, 'steps': [20, 30],
    'warmup_mode': 'linear', 'warmup_begin_lr': 0, 'warmup_steps': 10}
    >>> get_lr_scheduler("cosine", learning_rate=0.01, discount=0.1,
    ...     update_epoch=2, warmup_epoch=1, batches_per_epoch=10)  # doctest: +NORMALIZE_WHITESPACE
    {'scheduler': 'CosineScheduler', 'base_lr': 0.01, 'step': 1, 'max_steps': 20, 'final_lr': 0.001,
    'warmup_mode': 'linear', 'warmup_begin_lr': 0, 'warmup_steps': 10}
    >>> get_lr_scheduler("norm", learning_rate=0.01, discount=0.1,
    ...     update_epoch=2, warmup_epoch=1, batches_per_epoch=10)   # doctest: +NORMALIZE_WHITESPACE
    {'scheduler': 'NormScheduler', 'lr_loc': 0.005, 'step': 10, 'max_lr': 0.01, 'min_lr': 0.001,
    'lr_scale': 0.0025, 'warmup_mode': 'linear', 'warmup_begin_lr': 0, 'warmup_steps': 10}
    """
    if adaptive is True:
        for key in {"learning_rate", "lr"}:
            if key in kwargs:
                kwargs["base_lr"] = kwargs.pop(key)

        if isinstance(scheduler, str):
            scheduler = SCHEDULERS[scheduler].init(**kwargs)
    else:
        if isinstance(scheduler, str):
            scheduler = eval(scheduler)(**kwargs)

    assert isinstance(scheduler, LRScheduler)

    logger.info(scheduler)

    return scheduler


def plot_schedule(schedule_fn, iterations=1500, show=True):
    """

    Parameters
    ----------
    schedule_fn
    iterations
    show

    Returns
    -------
    learning rates: list

    Examples
    --------
    >>> scheduler = get_lr_scheduler("linear", base_lr=1, final_lr=0,
    ...     update_epoch=2, warmup_epoch=1, batches_per_epoch=10)
    >>> plot_schedule(scheduler, iterations=20, show=False)   # doctest: +NORMALIZE_WHITESPACE
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    >>> scheduler = get_lr_scheduler("norm", base_lr=1, discount=0.01,
    ...     batches_per_epoch=2, update_epoch=9, warmup_epoch=1)
    >>> plot_schedule(scheduler, iterations=20, show=False)   # doctest: +NORMALIZE_WHITESPACE
    [0.5, 1.0, 0.8051600872118375, 0.8051600872118375, 0.6911774184465967, 0.6911774184465967, 0.6076818248238643,
    0.6076818248238643, 0.5349275747204655, 0.5349275747204655, 0.46507242527953446, 0.46507242527953446,
    0.39231817517613554, 0.39231817517613554, 0.30882258155340314, 0.30882258155340314, 0.19483991278816226,
    0.19483991278816226, 0.01, 0.01]
    """
    import matplotlib.pyplot as plt

    # Iteration count starting at 1
    iterations = [i + 1 for i in range(iterations)]
    lrs = [schedule_fn(i) for i in iterations]

    plt.scatter(iterations, lrs)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    if show:  # pragma: no cover
        plt.show()
    return lrs


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    _scheduler = get_lr_scheduler(
        "norm", base_lr=1, discount=0.01, lr_loc=0.02,
        batches_per_epoch=100, update_epoch=10, epoch_update_freq=10,
    )
    plot_schedule(_scheduler)
