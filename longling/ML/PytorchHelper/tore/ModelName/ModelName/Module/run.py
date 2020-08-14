# coding: utf-8
# create by tongshiwei on 2019-9-1
from longling import path_append

try:
    # for python module
    from .etl import transform, etl, pseudo_data_iter
    from .configuration import Configuration, ConfigurationParser
    from .sym import get_net, get_bp_loss, fit_f, eval_f
except (ImportError, SystemError):  # pragma: no cover
    # for python script
    from etl import transform, etl, pseudo_data_iter
    from configuration import Configuration, ConfigurationParser
    from sym import get_net, get_bp_loss, fit_f, eval_f

from longling.ML.PytorchHelper import set_device


def numerical_check(_net, _cfg: Configuration, train_data, test_data, dump_result=False,
                    reporthook=None, final_reporthook=None):  # pragma: no cover
    ctx = _cfg.ctx

    _net = set_device(_net, ctx)

    bp_loss_f = get_bp_loss(ctx, **_cfg.loss_params)
    loss_function = {}
    loss_function.update(bp_loss_f)

    from longling.ML.toolkit import EvalFormatter as Formatter
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
    from longling.ML.PytorchHelper.toolkit.optimizer import get_trainer
    trainer = get_trainer(
        _net, optimizer=_cfg.optimizer,
        optimizer_params=_cfg.optimizer_params,
        select=_cfg.train_select
    )

    for epoch in range(_cfg.begin_epoch, _cfg.end_epoch):
        for batch_data in progress_monitor(train_data, "Epoch: %s" % epoch):
            fit_f(
                net=_net, batch_data=batch_data,
                trainer=trainer, bp_loss_f=bp_loss_f,
                loss_function=loss_function,
                loss_monitor=loss_monitor,
            )

        if epoch % 1 == 0:
            msg, data = evaluation_formatter(
                epoch=epoch,
                loss_name_value=dict(loss_monitor.items()),
                eval_name_value=eval_f(_net, test_data, ctx=ctx),
                extra_info=None,
                dump=dump_result,
            )
            print(msg)
            if reporthook is not None:
                reporthook(data)

    if final_reporthook is not None:
        final_reporthook()


def pseudo_numerical_check(_net, _cfg):  # pragma: no cover
    datas = pesudo_data_iter(_cfg)
    numerical_check(_net, _cfg, datas, datas, dump_result=False)


def train(train_fn, test_fn, reporthook=None, final_reporthook=None, **cfg_kwargs):  # pragma: no cover
    from longling.ML.toolkit.hyper_search import prepare_hyper_search

    cfg_kwargs, reporthook, final_reporthook, tag = prepare_hyper_search(
        cfg_kwargs, Configuration, reporthook, final_reporthook, primary_key="prf:avg:f1"
    )

    _cfg = Configuration(**cfg_kwargs)
    _net = get_net(**_cfg.hyper_params)

    train_data = etl(_cfg.var2val(train_fn), params=_cfg)
    test_data = etl(_cfg.var2val(test_fn), params=_cfg)

    numerical_check(_net, _cfg, train_data, test_data, dump_result=True)


def sym_run(stage: (int, str) = "viz"):  # pragma: no cover
    if isinstance(stage, str):
        stage = {
            "pseudo": 0,
            "real": 1,
            "cli": 2,
        }[stage]

    if stage == 0:
        # ############################## Pesudo Test #################################
        cfg = Configuration(
            hyper_params={
            },
            ctx="cuda:0,1,2",
        )
        net = get_net(**cfg.hyper_params)
        pseudo_numerical_check(net, cfg)

    elif stage == 1:
        # ################################# Simple Train ###############################
        train(
            "$data_dir/train",
            "$data_dir/test",
            dataset="",
            ctx="cuda:0",
            optimizer_params={
                "lr": 0.001
            },
            hyper_params={
            },
            batch_size=16,
        )

    elif stage == 2:
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


if __name__ == '__main__':
    sym_run("pseudo")
