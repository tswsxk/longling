# coding: utf-8
# 2021/2/10 @ tongshiwei

from longling import path_append
from longling.ML.toolkit.hyper_search import prepare_hyper_search
from longling.ML.toolkit import EpochEvalFMT as Formatter
from longling.ML.toolkit import MovingLoss
from longling.ML.toolkit import ConsoleProgressMonitor
from tqdm import tqdm
from longling.ML import get_epoch_params_filepath


def train(
        net, cfg, loss_function, trainer,
        train_data, test_data=None,
        params_save=False, dump_result=False, progress_monitor=None,
        *,
        fit_f, eval_f=None, net_init=None, get_net=None, get_loss=None, get_trainer=None, save_params=None,
        enable_hyper_search=False, reporthook=None, final_reporthook=None, primary_key=None,
        eval_epoch=1, loss_dict2tmt_loss=None,
        **cfg_kwargs):
    net = net if get_net is None else get_net(**cfg.hyper_params)

    if net_init is not None:
        net_init(net, cfg=cfg, **cfg.init_params)

    if enable_hyper_search:
        cfg_kwargs, reporthook, final_reporthook, tag = prepare_hyper_search(
            cfg_kwargs, reporthook, final_reporthook, primary_key=primary_key, with_keys="Epoch"
        )
        dump_result = not tag

    ctx = cfg.ctx
    batch_size = cfg.batch_size

    loss_function = get_loss(**cfg.loss_params) if get_loss is not None else loss_function

    if isinstance(loss_function, dict):
        _loss_function = loss_function
    else:
        if hasattr(loss_function, "__name__"):
            loss_name = loss_function.__name__
        elif hasattr(loss_function, "__class__"):
            loss_name = loss_function.__class__.__name__
        else:
            loss_name = "loss"
        loss_function = {loss_name: loss_function}
        if loss_dict2tmt_loss is not None:
            loss_function = loss_dict2tmt_loss(loss_function)
        _loss_function = list(loss_function.values())[0]

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
            total_epoch=cfg.end_epoch - 1
        )
    elif progress_monitor is None or progress_monitor == "tqdm":
        def progress_monitor(x, e):
            return tqdm(x, "Epoch: %s" % e)

    if dump_result:
        from longling import config_logging
        validation_logger = config_logging(
            filename=path_append(cfg.model_dir, "result.log"),
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
    trainer = get_trainer(
        net, optimizer=cfg.optimizer,
        optimizer_params=cfg.optimizer_params,
        select=cfg.train_select,
        lr_params=cfg.lr_params
    ) if get_trainer is not None else trainer

    for epoch in range(cfg.begin_epoch, cfg.end_epoch):
        for i, batch_data in enumerate(progress_monitor(train_data, epoch)):
            fit_f(
                net, batch_size=batch_size, batch_data=batch_data,
                trainer=trainer,
                loss_function=_loss_function,
                loss_monitor=loss_monitor,
                ctx=ctx,
            )

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
                eval_name_value=eval_f(net, test_data, ctx=ctx),
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

    if final_reporthook is not None:
        final_reporthook()
