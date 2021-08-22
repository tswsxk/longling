# coding: utf-8
# 2021/2/12 @ tongshiwei

from longling.ML.DL import light_module as dlm
from .toolkit import loss_dict2tmt_torch_loss, get_trainer as _get_trainer, save_params as _save_params
from .tore import net_init as _net_init


def train(
        net, cfg, loss_function, trainer,
        train_data, test_data=None,
        params_save=False, dump_result=False, progress_monitor=None,
        *,
        fit_f, eval_f=None, get_net=None, get_loss=None, get_trainer=None, save_params=None,
        enable_hyper_search=False, reporthook=None, final_reporthook=None, primary_key=None,
        eval_epoch=1, initial_net=True, net_init=None, batch_lr_scheduler=None, epoch_lr_scheduler=None,
        loss_as_dict=False, verbose=None, dump_cfg=None, **cfg_kwargs
):
    if initial_net:
        net_init = _net_init if net_init is None else net_init
    else:
        net_init = None

    if trainer is None and get_trainer is None:
        get_trainer = _get_trainer

    if cfg.lr_params:
        assert batch_lr_scheduler or epoch_lr_scheduler

    save_params = _save_params if save_params is None else save_params

    dlm.train(
        net, cfg, loss_function, trainer, train_data, test_data, params_save, dump_result, progress_monitor,
        fit_f=fit_f, eval_f=eval_f,
        net_init=net_init, get_net=get_net, get_loss=get_loss, get_trainer=get_trainer,
        save_params=save_params,
        enable_hyper_search=enable_hyper_search, reporthook=reporthook,
        final_reporthook=final_reporthook,
        primary_key=primary_key, eval_epoch=eval_epoch, loss_dict2tmt_loss=loss_dict2tmt_torch_loss,
        batch_lr_scheduler=batch_lr_scheduler, epoch_lr_scheduler=epoch_lr_scheduler, loss_as_dict=loss_as_dict,
        verbose=verbose, dump_cfg=dump_cfg, **cfg_kwargs
    )
