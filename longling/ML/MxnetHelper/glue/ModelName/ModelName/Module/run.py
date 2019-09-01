# coding: utf-8
# create by tongshiwei on 2019-9-1
from longling import path_append

try:
    # for python module
    from .sym import NetName, BP_LOSS_F, fit_f, eval_f, net_viz
    from .etl import transform, etl
    from .configuration import Configuration, ConfigurationParser
except (ImportError, SystemError):
    # for python script
    from sym import NetName, BP_LOSS_F, fit_f, eval_f, net_viz
    from etl import transform, etl
    from configuration import Configuration, ConfigurationParser


# todo:
def get_data_iter(_cfg):
    def pseudo_data_generation():
        # 在这里定义测试用伪数据流
        import random
        random.seed(10)

        raw_data = [
            [random.random() for _ in range(5)]
            for _ in range(1000)
        ]

        return raw_data

    return transform(pseudo_data_generation(), _cfg)


def numerical_check(_net, _cfg: Configuration, train_data, test_data, dump_result=False):
    ctx = _cfg.ctx
    batch_size = _cfg.batch_size

    _net.initialize(ctx)

    bp_loss_f = BP_LOSS_F
    loss_function = {}
    loss_function.update(bp_loss_f)

    from longling.ML.MxnetHelper.glue import module
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
    trainer = module.Module.get_trainer(
        _net, optimizer=_cfg.optimizer,
        optimizer_params=_cfg.optimizer_params,
        select=_cfg.train_select
    )

    for epoch in range(_cfg.begin_epoch, _cfg.end_epoch):
        for batch_data in progress_monitor(train_data, "Epoch: %s" % epoch):
            fit_f(
                net=_net, batch_size=batch_size, batch_data=batch_data,
                trainer=trainer, bp_loss_f=bp_loss_f,
                loss_function=loss_function,
                loss_monitor=loss_monitor,
                ctx=ctx,
            )

        print("epoch-%d: %s" % (epoch, list(loss_monitor.items())))

        if epoch % 1 == 0:
            if epoch % 1 == 0:
                print(
                    evaluation_formatter(
                        epoch=epoch,
                        loss_name_value=dict(loss_monitor.items()),
                        eval_name_value=eval_f(_net, test_data, ctx=ctx),
                        extra_info=None,
                        dump=True,
                    )[0]
                )


def pesudo_numerical_check(_net, _cfg):
    datas = get_data_iter(_cfg)
    numerical_check(_net, _cfg, datas, datas, dump_result=False)


def train(train_fn, test_fn, **cfg_kwargs):
    _cfg = Configuration(**cfg_kwargs)
    _net = NetName(**_cfg.hyper_params)

    train_data = etl(_cfg.var2val(train_fn), params=_cfg)
    test_data = etl(_cfg.var2val(test_fn), params=_cfg)

    numerical_check(_net, _cfg, train_data, test_data, dump_result=True)


def sym_run(stage: (int, str) = "viz"):
    if isinstance(stage, str):
        stage = {
            "viz": 0,
            "pesudo": 1,
            "real": 2,
            "cli": 3,
        }[stage]

    if stage <= 1:
        cfg = Configuration()
        net = NetName()

        if stage == 0:
            # ############################## Net Visualization ###########################
            net_viz(net, cfg, False)
        else:
            # ############################## Pesudo Test #################################
            pesudo_numerical_check(net, cfg)

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
            hyper_params={
            },
            batch_size=16,
        )

    elif stage == 3:
        # ################################# CLI ###########################
        cfg_parser = ConfigurationParser(Configuration)
        cfg_parser.add_subcommand(cfg_parser.func_spec(train))
        cfg_kwargs = cfg_parser()
        assert "subcommand" in cfg_kwargs
        subcommand = cfg_kwargs["subcommand"]
        del cfg_kwargs["subcommand"]
        print(cfg_kwargs)
        eval("%s" % subcommand)(**cfg_kwargs)

    else:
        raise TypeError


if __name__ == '__main__':
    sym_run("viz")