# coding: utf-8
# 2021/2/10 @ tongshiwei

from longling import path_append
from longling.ML.toolkit.hyper_search import prepare_hyper_search
from longling.ML.toolkit import EpochEvalFMT as Formatter
from longling.ML.toolkit import MovingLoss
from longling.ML.toolkit import ConsoleProgressMonitor
from tqdm import tqdm
from longling.ML import get_epoch_params_filepath
from longling.ML.const import RESULT_LOG


def train(
        net, cfg, loss_function, trainer,
        train_data, test_data=None,
        params_save=False, dump_result=False, progress_monitor=None,
        *,
        fit_f, eval_f=None, net_init=None, get_net=None, get_loss=None, get_trainer=None, save_params=None,
        enable_hyper_search=False, reporthook=None, final_reporthook=None, primary_key=None,
        eval_epoch=1, loss_dict2tmt_loss=None, epoch_lr_scheduler=None, batch_lr_scheduler=None, loss_as_dict=False,
        verbose=None, dump_cfg=None, **cfg_kwargs):
    if enable_hyper_search:
        assert get_net is not None
        cfg_kwargs, reporthook, final_reporthook, tag = prepare_hyper_search(
            cfg_kwargs, reporthook, final_reporthook, primary_key=primary_key, with_keys="Epoch", dump=params_save
        )
        dump_result = tag
        verbose = tag if verbose is None else verbose
        cfg.update(**cfg_kwargs)
        print("hyper search enabled")
        print(cfg)

    verbose = True if verbose is None else verbose
    dump_cfg = dump_cfg if dump_cfg is not None else params_save
    if dump_cfg:
        cfg.dump()

    net = net if get_net is None else get_net(**cfg.hyper_params)

    if net_init is not None:
        net_init(net, cfg=cfg, initializer_kwargs=cfg.init_params)

    train_ctx = cfg.ctx if cfg.train_ctx is None else cfg.train_ctx
    eval_ctx = cfg.ctx if cfg.eval_ctx is None else cfg.eval_ctx
    batch_size = cfg.batch_size

    loss_function = get_loss(**cfg.loss_params) if get_loss is not None else loss_function

    if isinstance(loss_function, dict):
        _loss_function = loss_function
    else:
        if hasattr(loss_function, "__name__"):
            loss_name = loss_function.__name__
        elif hasattr(loss_function, "__class__"):
            loss_name = loss_function.__class__.__name__
        else:  # pragma: no cover
            loss_name = "loss"
        loss_function = {loss_name: loss_function}
        if loss_dict2tmt_loss is not None:
            loss_function = loss_dict2tmt_loss(loss_function)
        _loss_function = list(loss_function.values())[0] if loss_as_dict is False else loss_function

    loss_monitor = MovingLoss(loss_function)

    if progress_monitor is None and loss_dict2tmt_loss is not None:
        progress_monitor = ConsoleProgressMonitor(
            indexes={
                "Loss": [name for name in loss_function]
            },
            values={
                "Loss": loss_monitor.losses
            },
            player_type="epoch",
            total_epoch=cfg.end_epoch - 1,
            silent=not verbose
        )
    elif progress_monitor is None or progress_monitor == "tqdm":
        def progress_monitor(x, e):
            return tqdm(x, "Epoch: %s" % e, disable=not verbose)

    if dump_result:
        from longling import config_logging
        validation_logger = config_logging(
            filename=path_append(cfg.model_dir, cfg.get("result_log", RESULT_LOG)),
            logger="%s-validation" % cfg.model_name,
            mode="w",
            log_format="%(message)s",
        )
        evaluation_formatter = Formatter(
            logger=validation_logger,
            dump_file=cfg.validation_result_file,
        )
    else:
        evaluation_formatter = Formatter()

    # train check
    if get_trainer is not None:
        trainer = get_trainer(
            net, optimizer=cfg.optimizer,
            optimizer_params=cfg.optimizer_params,
            select=cfg.train_select,
            lr_params=cfg.lr_params
        )
        if batch_lr_scheduler is True:
            trainer, batch_lr_scheduler = trainer
        elif epoch_lr_scheduler is True:
            trainer, epoch_lr_scheduler = trainer

    for epoch in range(cfg.begin_epoch, cfg.end_epoch):
        for i, batch_data in enumerate(progress_monitor(train_data, epoch)):
            fit_f(
                net, batch_size=batch_size, batch_data=batch_data,
                trainer=trainer,
                loss_function=_loss_function,
                loss_monitor=loss_monitor,
                ctx=train_ctx,
            )
            if batch_lr_scheduler is not None:
                batch_lr_scheduler.step()

        if cfg.lr_params and "update_params" in cfg.lr_params and cfg.end_epoch - cfg.begin_epoch - 1 > 0:
            cfg.logger.info("reset trainer")
            lr_params = cfg.lr_params.pop("update_params")
            lr_update_params = dict(
                batches_per_epoch=i + 1,
                lr=cfg.optimizer_params["learning_rate"],
                update_epoch=lr_params.get(
                    "update_epoch",
                    cfg.end_epoch - cfg.begin_epoch - 1
                )
            )
            lr_update_params.update(lr_params)

            assert get_trainer is not None
            trainer = get_trainer(
                net, optimizer=cfg.optimizer,
                optimizer_params=cfg.optimizer_params,
                lr_params=lr_update_params,
                select=cfg.train_select,
                logger=cfg.logger
            )

        if test_data is not None and epoch % eval_epoch == 0:
            msg, data = evaluation_formatter(
                iteration=epoch,
                loss_name_value=dict(loss_monitor.items()),
                eval_name_value=eval_f(net, test_data, ctx=eval_ctx, verbose=verbose, **cfg.get("eval_params", {})),
                extra_info=None,
                dump=dump_result,
                keep={"msg", "data"}
            )
            print(msg)
            if reporthook is not None:
                reporthook(data)

        # optional
        loss_monitor.reset()

        if params_save and (epoch % cfg.save_epoch == 0 or epoch == cfg.end_epoch - 1):
            assert save_params is not None
            params_path = get_epoch_params_filepath(cfg.model_name, epoch, cfg.model_dir)
            cfg.logger.info("save model params to %s, with select='%s'" % (params_path, cfg.save_select))
            save_params(params_path, net, select=cfg.save_select)

        if epoch_lr_scheduler is not None:
            epoch_lr_scheduler.step()

    if final_reporthook is not None:
        final_reporthook()
