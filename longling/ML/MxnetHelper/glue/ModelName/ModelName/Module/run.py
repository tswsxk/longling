# coding: utf-8
# create by tongshiwei on 2019-9-1
from longling import path_append
from longling.ML import get_epoch_params_filepath
from longling.ML.MxnetHelper import save_params

try:
    # for python module
    from .sym import get_net, get_loss, fit_f, eval_f, net_viz, net_init
    from .etl import transform, etl, pseudo_data_iter
    from .configuration import Configuration, ConfigurationParser
except (ImportError, SystemError):  # pragma: no cover
    # for python script
    from sym import get_net, get_loss, fit_f, eval_f, net_viz, net_init
    from etl import transform, etl, pseudo_data_iter
    from configuration import Configuration, ConfigurationParser


def numerical_check(_net, _cfg: Configuration, train_data, test_data, dump_result=False,
                    reporthook=None, final_reporthook=None, params_save=False):  # pragma: no cover
    ctx = _cfg.ctx
    batch_size = _cfg.batch_size

    loss_function = get_loss(**_cfg.loss_params)

    from longling.ML.MxnetHelper.glue import module
    from longling.ML.toolkit import EpochEvalFMT as Formatter
    from longling.ML.toolkit import MovingLoss
    from tqdm import tqdm

    loss_monitor = MovingLoss(loss_function)
    progress_monitor = tqdm
    if dump_result:
        from longling import config_logging
        validation_logger = config_logging(
            filename=path_append(_cfg.model_dir, "result.log"),
            logger="%s-validation" % _cfg.model_name,
            mode="w",
            log_format="%(message)s",
        )
        evaluation_formatter = Formatter(
            logger=validation_logger,
            dump_file=_cfg.validation_result_file,
        )
    else:
        evaluation_formatter = Formatter()

    # train check
    trainer = module.Module.get_trainer(
        _net, optimizer=_cfg.optimizer,
        optimizer_params=_cfg.optimizer_params,
        select=_cfg.train_select,
        lr_params=_cfg.lr_params
    )

    for epoch in range(_cfg.begin_epoch, _cfg.end_epoch):
        for i, batch_data in enumerate(progress_monitor(train_data, "Epoch: %s" % epoch)):
            fit_f(
                net=_net, batch_size=batch_size, batch_data=batch_data,
                trainer=trainer,
                loss_function=loss_function,
                loss_monitor=loss_monitor,
                ctx=ctx,
            )
        if _cfg.lr_params and "update_params" in _cfg.lr_params and _cfg.end_epoch - _cfg.begin_epoch - 1 > 0:
            _cfg.logger.info("reset trainer")
            lr_params = _cfg.lr_params.pop("update_params")
            lr_update_params = dict(
                batches_per_epoch=i + 1,
                lr=_cfg.optimizer_params["learning_rate"],
                update_epoch=lr_params.get(
                    "update_epoch",
                    _cfg.end_epoch - _cfg.begin_epoch - 1
                )
            )
            lr_update_params.update(lr_params)

            trainer = module.Module.get_trainer(
                _net, optimizer=_cfg.optimizer,
                optimizer_params=_cfg.optimizer_params,
                lr_params=lr_update_params,
                select=_cfg.train_select,
                logger=_cfg.logger
            )

        if epoch % 1 == 0:
            msg, data = evaluation_formatter(
                iteration=epoch,
                loss_name_value=dict(loss_monitor.items()),
                eval_name_value=eval_f(_net, test_data, ctx=ctx),
                extra_info=None,
                dump=dump_result,
                keep={"msg", "data"}
            )
            print(msg)
            if reporthook is not None:
                reporthook(data)

        # optional
        loss_monitor.reset()

        if params_save and (epoch % _cfg.save_epoch == 0 or epoch == _cfg.end_epoch - 1):
            params_path = get_epoch_params_filepath(_cfg.model_name, epoch, _cfg.model_dir)
            _cfg.logger.info("save model params to %s, with select='%s'" % (params_path, _cfg.save_select))
            save_params(params_path, _net, select=_cfg.save_select)

    if final_reporthook is not None:
        final_reporthook()


def pseudo_numerical_check(_net, _cfg):  # pragma: no cover
    datas = pseudo_data_iter(_cfg)
    numerical_check(_net, _cfg, datas, datas, dump_result=False)


def train(train_fn, test_fn, reporthook=None, final_reporthook=None,
          primary_key="macro_avg:f1", params_save=False, **cfg_kwargs):  # pragma: no cover
    from longling.ML.toolkit.hyper_search import prepare_hyper_search

    cfg_kwargs, reporthook, final_reporthook, tag = prepare_hyper_search(
        cfg_kwargs, reporthook, final_reporthook, primary_key=primary_key, with_keys="Epoch"
    )

    _cfg = Configuration(**cfg_kwargs)
    print(_cfg)
    _net = get_net(**_cfg.hyper_params)
    net_init(_net, cfg=_cfg, **_cfg.init_params)

    train_data = etl(_cfg.var2val(train_fn), params=_cfg)
    test_data = etl(_cfg.var2val(test_fn), params=_cfg)

    numerical_check(_net, _cfg, train_data, test_data, dump_result=not tag, reporthook=reporthook,
                    final_reporthook=final_reporthook, params_save=params_save)


def sym_run(stage: (int, str) = "viz"):  # pragma: no cover
    if isinstance(stage, str):
        stage = {
            "viz": 0,
            "pseudo": 1,
            "real": 2,
            "cli": 3,
        }[stage]

    if stage <= 1:
        cfg = Configuration(
            hyper_params={}
        )
        net = get_net(**cfg.hyper_params)
        net.initialize()

        if stage == 0:
            # ############################## Net Visualization ###########################
            net_viz(net, cfg, False)
        else:
            # ############################## Pseudo Test #################################
            pseudo_numerical_check(net, cfg)

    elif stage == 2:
        # ################################# Simple Train ###############################
        import mxnet as mx
        train(
            "$data_dir/train",
            "$data_dir/test",
            ctx=mx.cpu(),
            optimizer_params={
                "learning_rate": 0.001
            },
            hyper_params_update={
            },
            batch_size=16,
            params_save=False
        )

    elif stage == 3:
        # ################################# CLI ###########################
        cfg_parser = ConfigurationParser(Configuration, commands=[train])
        cfg_kwargs = cfg_parser()
        assert "subcommand" in cfg_kwargs
        subcommand = cfg_kwargs["subcommand"]
        del cfg_kwargs["subcommand"]
        print(cfg_kwargs)
        eval("%s" % subcommand)(**cfg_kwargs)

    else:
        raise TypeError


if __name__ == '__main__':  # pragma: no cover
    sym_run("viz")
