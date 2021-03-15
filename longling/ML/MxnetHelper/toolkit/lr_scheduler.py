# coding: utf-8
# 2020/8/18 @ tongshiwei

import logging
import numpy as np
import scipy.stats as st
from mxnet.lr_scheduler import LRScheduler, FactorScheduler, PolyScheduler, MultiFactorScheduler, CosineScheduler

from longling.ML.DL import get_total_update_steps, get_factor_lr_params, stop_lr as _stop_lr

__all__ = ["get_lr_scheduler", "plot_schedule", "get_total_update_steps"]


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
            final_lr, discount = _stop_lr(base_lr, discount=discount)
            kwargs.update({"final_lr": final_lr})

        return kwargs


class _FactorScheduler(FactorScheduler, _LRScheduler):
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
        kwargs = super(_FactorScheduler, cls).init(
            base_lr,
            warmup_epoch=warmup_epoch,
            batches_per_epoch=batches_per_epoch,
            discount=discount
        )

        if batches_per_epoch is not None and update_epoch is not None:
            step, factor, stop_factor_lr = get_factor_lr_params(
                base_lr,
                update_epoch,
                batches_per_epoch,
                epoch_update_freq,
                kwargs.get("final_lr", kwargs.get("stop_factor_lr", None)),
                discount,
            )
            kwargs.update({"step": step, "factor": factor, "stop_factor_lr": stop_factor_lr})

        return cls(base_lr=base_lr, **kwargs)


class _PolyScheduler(PolyScheduler, _LRScheduler):
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
    def init(cls, base_lr, batches_per_epoch=None, update_epoch=None, epoch_update_freq=1, warmup_epoch=0,
             discount=None, *args, **kwargs):
        kwargs = super(_PolyScheduler, cls).init(
            base_lr,
            warmup_epoch=warmup_epoch,
            batches_per_epoch=batches_per_epoch,
            discount=discount
        )

        if batches_per_epoch is not None and update_epoch is not None:
            max_update = batches_per_epoch * update_epoch + kwargs.get("warmup_steps", 0)
            kwargs.update({"max_update": max_update})

        return cls(base_lr=base_lr, **kwargs)


class _CosineScheduler(CosineScheduler, _LRScheduler):
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
    def init(cls, base_lr, batches_per_epoch=None, update_epoch=None, epoch_update_freq=1, warmup_epoch=0,
             discount=None, *args, **kwargs):
        kwargs.update(super(_CosineScheduler, cls).init(
            base_lr,
            warmup_epoch=warmup_epoch,
            batches_per_epoch=batches_per_epoch,
            discount=discount
        ))

        if batches_per_epoch is not None and update_epoch is not None:
            max_update = batches_per_epoch * update_epoch * epoch_update_freq + kwargs.get("warmup_steps", 0)
            kwargs.update({"max_update": max_update})

        return cls(base_lr=base_lr, **kwargs)


class _MultiFactorScheduler(MultiFactorScheduler, _LRScheduler):
    @property
    def state_dict(self):
        return {
            "scheduler": "MultiFactorScheduler",
            "base_lr": self.base_lr,
            "max_steps": self.step,
            "warmup_mode": self.warmup_mode,
            "warmup_begin_lr": self.warmup_begin_lr,
            "warmup_steps": self.warmup_steps,
        }

    @classmethod
    def init(cls, base_lr, batches_per_epoch=None, update_epoch=None, epoch_update_freq=1, warmup_epoch=0,
             discount=None, *args, **kwargs):
        kwargs = super(_MultiFactorScheduler, cls).init(
            base_lr,
            warmup_epoch=warmup_epoch,
            batches_per_epoch=batches_per_epoch,
            discount=discount
        )
        return cls(base_lr=base_lr, **kwargs)


class LinearScheduler(LRScheduler):
    def __init__(self, max_update, base_lr=0.01, final_lr=0,
                 warmup_steps=0, warmup_begin_lr=0, warmup_mode='linear'):
        super(LinearScheduler, self).__init__(
            base_lr, warmup_steps=warmup_steps, warmup_begin_lr=warmup_begin_lr, warmup_mode=warmup_mode
        )
        assert isinstance(max_update, int)
        if max_update < 1:
            raise ValueError("maximum number of updates must be strictly positive")
        self.max_update = max_update
        self.final_lr = final_lr
        self.max_steps = self.max_update - self.warmup_steps
        self.increase = (self.final_lr - self.base_lr) / float(self.max_steps)

    def __call__(self, num_update):
        if num_update < self.warmup_steps:
            return self.get_warmup_lr(num_update)
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
        kwargs = super(_LinearScheduler, cls).init(
            base_lr,
            warmup_epoch=warmup_epoch,
            batches_per_epoch=batches_per_epoch,
            discount=discount
        )
        if batches_per_epoch is not None and update_epoch is not None:
            max_update = batches_per_epoch * update_epoch * epoch_update_freq + kwargs.get("warmup_steps", 0)
            kwargs.update({"max_update": max_update})
        return cls(base_lr=base_lr, **kwargs)


class NormScheduler(LRScheduler):
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
        self.p_increase = self.step / (max_update - warmup_steps)
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
        while num_update - self.warmup_steps > self.count + self.step:
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
        kwargs.update(super(_NormScheduler, cls).init(
            base_lr,
            warmup_epoch=warmup_epoch,
            batches_per_epoch=batches_per_epoch,
            discount=discount,
        ))
        if batches_per_epoch is not None and update_epoch is not None:
            max_update = int(batches_per_epoch * update_epoch) + kwargs.get("warmup_steps", 0)
            kwargs.update({"max_update": max_update, "step": batches_per_epoch // epoch_update_freq})
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


def plot_schedule(schedule_fn, iterations=1500):
    import matplotlib.pyplot as plt
    # Iteration count starting at 1
    iterations = [i + 1 for i in range(iterations)]
    lrs = [schedule_fn(i) for i in iterations]
    plt.scatter(iterations, lrs)
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.show()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    _scheduler = get_lr_scheduler(
        "norm", base_lr=1, discount=0.01, lr_loc=0.02,
        batches_per_epoch=100, update_epoch=10, epoch_update_freq=10,
    )
    plot_schedule(_scheduler)
