# coding: utf-8
# Copyright @tongshiwei

import torch

try:
    from .Module import *
except (SystemError, ModuleNotFoundError):
    from Module import *


class ModelName(object):
    def __init__(
            self, load_epoch=None, cfg=None, toolbox_init=False, **kwargs
    ):
        """
        模型初始化

        Parameters
        ----------
        load_epoch: int or None
            默认为 None，不为None时，将装载指定轮数的模型参数作为初始化模型参数
        cfg: Configuration, str or None
            默认为 None，不为None时，将使用cfg指定的参数配置
            或路径指定的参数配置作为模型参数
        toolbox_init: bool
            默认为 False，是否初始化工具包
        kwargs
            参数配置可选参数
        """
        # 1 配置参数初始化
        # todo 到Configuration处定义相关参数
        cfg = self.config(cfg, **kwargs)
        self.mod = self.get_module(cfg)

        # 2 todo 定义网络结构
        # 2.1 重新生成
        self.mod.logger.info("generating symbol")
        net = self.mod.sym_gen(
            **cfg.hyper_params
        )
        # 2.2 装载已有模型, export 出来的文件
        # net = mod.load(begin_epoch)
        # net = GluonModule.load_net(filename)

        self.net = net
        self.initialized = False

        if load_epoch is not None:
            self.model_init(load_epoch, allow_reinit=False)

        self.bp_loss_f = None
        self.loss_function = None
        self.toolbox = None
        self.trainer = None

        if toolbox_init:
            self.toolbox_init(**self.mod.cfg.toolbox_params)

    @staticmethod
    def config(cfg=None, **kwargs):
        """
        配置初始化

        Parameters
        ----------
        cfg: Configuration, str or None
            默认为 None，不为None时，将使用cfg指定的参数配置
            或路径指定的参数配置作为模型参数
        kwargs
            参数配置可选参数
        """
        cfg = Configuration(
            **kwargs
        ) if cfg is None else cfg
        if not isinstance(cfg, Configuration):
            cfg = Configuration.load_cfg(cfg, **kwargs)
        cfg.dump(override=True)
        return cfg

    @staticmethod
    def get_module(cfg):
        """
        根据配置，生成模型模块

        Parameters
        ----------
        cfg: Configuration
            模型配置参数
        Returns
        -------
        mod: Module
            模型模块
        """
        mod = Module(cfg)
        mod.logger.info(str(mod))
        filename = mod.cfg.cfg_path
        mod.logger.info("parameters saved to %s" % filename)
        return mod

    def set_loss(self, bp_loss_f=None, loss_function=None):
        bp_loss_f = get_bp_loss(self.mod.cfg.ctx, **self.mod.cfg.loss_params) if bp_loss_f is None else bp_loss_f

        assert bp_loss_f is not None and len(bp_loss_f) == 1

        loss_function = {

        } if loss_function is None else loss_function
        loss_function.update(bp_loss_f)

        self.bp_loss_f = bp_loss_f
        self.loss_function = loss_function

    def toolbox_init(
            self,
            evaluation_formatter_parameters=None,
            validation_logger_mode="w",
            silent=False,
    ):

        from longling import path_append
        from longling.lib.clock import Clock
        from longling.lib.utilog import config_logging
        from longling.ML.toolkit import EpochEvalFMT as Formatter
        from longling.ML.toolkit import MovingLoss, ConsoleProgressMonitor as ProgressMonitor

        self.toolbox = {
            "monitor": dict(),
            "timer": None,
            "formatter": dict(),
        }

        mod = self.mod
        cfg = self.mod.cfg

        # 4.1 todo 定义损失函数
        # bp_loss_f 定义了用来进行 back propagation 的损失函数，
        # 有且只能有一个，命名中不能为 *_\d+ 型

        assert self.loss_function is not None

        loss_monitor = MovingLoss(self.loss_function)

        # 4.1 todo 初始化一些训练过程中的交互信息
        timer = Clock()

        progress_monitor = ProgressMonitor(
            indexes={
                "Loss": [name for name in self.loss_function]
            },
            values={
                "Loss": loss_monitor.losses
            },
            end_epoch=cfg.end_epoch - 1,
            silent=silent
        )

        validation_logger = config_logging(
            filename=path_append(cfg.model_dir, "result.log"),
            logger="%s-validation" % cfg.model_name,
            mode=validation_logger_mode,
            log_format="%(message)s",
        )

        # set evaluation formatter
        evaluation_formatter_parameters = {} \
            if evaluation_formatter_parameters is None \
            else evaluation_formatter_parameters

        evaluation_formatter = Formatter(
            logger=validation_logger,
            dump_file=mod.cfg.validation_result_file,
            **evaluation_formatter_parameters
        )

        self.toolbox["monitor"]["loss"] = loss_monitor
        self.toolbox["monitor"]["progress"] = progress_monitor
        self.toolbox["timer"] = timer
        self.toolbox["formatter"]["evaluation"] = evaluation_formatter

    def model_init(
            self,
            load_epoch=None, force_init=False, cfg=None,
            allow_reinit=True, trainer=None,
            **kwargs
    ):
        mod = self.mod
        net = self.net
        cfg = mod.cfg if cfg is None else cfg
        begin_epoch = cfg.begin_epoch

        if self.initialized and not force_init:
            mod.logger.warning("model has been initialized, skip model_init")

        load_epoch = load_epoch if load_epoch is not None else begin_epoch

        # 5 todo 初始化模型
        model_file = kwargs.get(
            "init_model_file", mod.epoch_params_filename(load_epoch)
        )
        try:
            net = mod.load(load_epoch)
            mod.logger.info(
                "load params from existing model file "
                "%s" % model_file
            )
        except FileExistsError:
            if allow_reinit:
                mod.logger.info("model doesn't exist, initializing")
                self.Module.net_initialize(net, cfg.ctx)
            else:
                mod.logger.info(
                    "model doesn't exist, target file: %s" % model_file
                )

        self.initialized = True

        # # optional
        # # todo: whether to use static symbol to accelerate
        # # note: do not invoke this method for dynamic structure like rnn
        # # suggestion: annotate this until your process worked
        # net.hybridize()

        self.trainer = Module.get_trainer(
            net, optimizer=cfg.optimizer,
            optimizer_params=cfg.optimizer_params,
            lr_params=cfg.lr_params,
            select=cfg.train_select
        ) if trainer is None else trainer

    def train_net(self, train_data, eval_data, trainer=None):
        mod = self.mod
        cfg = self.mod.cfg
        net = self.net

        bp_loss_f = self.bp_loss_f
        loss_function = self.loss_function
        toolbox = self.toolbox

        batch_size = mod.cfg.batch_size
        begin_epoch = mod.cfg.begin_epoch
        end_epoch = mod.cfg.end_epoch

        assert all([bp_loss_f, loss_function]), \
            "make sure these variable have been initialized, " \
            "check init method and " \
            "make sure package_init method has been called"

        # 6 todo 训练
        trainer = self.trainer if trainer is None else trainer
        mod.logger.info("start training")
        mod.fit(
            net=net, begin_epoch=begin_epoch, end_epoch=end_epoch,
            batch_size=batch_size,
            train_data=train_data,
            trainer=trainer, bp_loss_f=bp_loss_f,
            loss_function=loss_function,
            eval_data=eval_data,
            ctx=None,
            toolbox=toolbox,
            prefix=mod.prefix,
            save_epoch=cfg.save_epoch,
        )

        # optional
        # # need execute the net.hybridize() before export,
        # # and execute forward at least one time
        # 需要在这之前调用 hybridize 方法,并至少forward一次
        # # net.export(mod.prefix)

    def step_fit(self, batch_data, trainer=None, loss_monitor=None):
        mod = self.mod
        net = self.net

        bp_loss_f = self.bp_loss_f
        loss_function = self.loss_function

        if loss_monitor is None:
            toolbox = self.toolbox
            monitor = toolbox.get("monitor")
            loss_monitor = monitor.get("loss") if monitor else None

        trainer = self.trainer if trainer is None else trainer
        mod.fit_f(
            net=net, batch_data=batch_data,
            trainer=trainer,
            bp_loss_f=bp_loss_f, loss_function=loss_function,
            loss_monitor=loss_monitor,
        )

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def call(self, x, ctx=None):
        # call forward for single data

        # optional
        # pre process x

        # convert the data to ndarray
        x = torch.tensor([x])

        # forward
        outputs = self.net(x).asnumpy().tolist()[0]

        return outputs

    def batch_call(self, x):
        # call forward for batch data
        # notice: do not use too big batch size

        # optional
        # pre process x

        # convert the data to ndarray
        x = torch.tensor(x)

        # forward
        outputs = self.net(x).asnumpy().tolist()

        return outputs

    def transform(self, data):
        return transform(data, self.mod.cfg)

    def etl(self, data_src, cfg=None):
        mod = self.mod
        cfg = mod.cfg if cfg is None else cfg

        # 4.2 todo 定义数据加载
        mod.logger.info("loading data")
        data = etl(cfg.var2val(data_src), params=cfg)

        return data

    def _train(self, train, valid):
        train_data = self.etl(train)
        valid_data = self.etl(valid)
        self.train_net(train_data, valid_data)

    @staticmethod
    def train(*args, cfg=None, **kwargs):
        module = ModelName(cfg=cfg, **kwargs)
        module.set_loss()
        # module.viz()

        module.toolbox_init()
        module.model_init(**kwargs)

        module._train(*args)

        return module

    @staticmethod
    def test(test_filename, test_epoch, dump_file=None, **kwargs):
        from longling.ML.toolkit.formatter import EpochEvalFMT as Formatter
        formatter = Formatter(dump_file=dump_file)
        module = ModelName.load(test_epoch, **kwargs)

        test_data = module.etl(test_filename)
        eval_result = module.mod.eval(module.net, test_data)
        formatter(
            tips="test",
            eval_name_value=eval_result
        )
        return eval_result

    @staticmethod
    def inc_train(init_model_file, *args, validation_logger_mode="w", **kwargs):
        # 增量学习，从某一轮参数继续训练
        module = ModelName(**kwargs)
        module.toolbox_init(validation_logger_mode=validation_logger_mode)
        module.model_init(init_model_file=init_model_file)

        module._train(*args)

    @staticmethod
    def dump_configuration(**kwargs):
        ModelName.get_module(**kwargs)

    @staticmethod
    def load(load_epoch=None, **kwargs):
        module = ModelName(**kwargs)
        load_epoch = module.mod.cfg.end_epoch if load_epoch is None \
            else load_epoch
        module.model_init(load_epoch, **kwargs)
        return module

    @staticmethod
    def get_parser():
        cfg_parser = ConfigurationParser(
            Configuration,
            commands=[
                ModelName.config,
                ModelName.train, ModelName.test,
                ModelName.inc_train,
            ]
        )
        return cfg_parser

    @staticmethod
    def run(parse_args=None):
        cfg_parser = ModelName.get_parser()
        cfg_kwargs = cfg_parser(parse_args)

        if "subcommand" not in cfg_kwargs:
            cfg_parser.print_help()
            return
        subcommand = cfg_kwargs["subcommand"]
        del cfg_kwargs["subcommand"]

        eval("%s.%s" % (ModelName.__name__, subcommand))(**cfg_kwargs)


if __name__ == '__main__':
    ModelName.run()
