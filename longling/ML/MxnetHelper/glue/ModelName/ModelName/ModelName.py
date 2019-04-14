# coding: utf-8
# Copyright @tongshiwei
import mxnet as mx

try:
    from .Module import *
except (SystemError, ModuleNotFoundError):
    from Module import *


class ModelName(object):
    def __init__(
            self, load_epoch=None, cfg=None, toolbox_init=False, **kwargs
    ):
        # 1 配置参数初始化
        cfg = self.config(cfg, **kwargs)
        self.mod = self.get_module(cfg)

        # 2 todo 定义网络结构
        # 2.1 重新生成
        self.mod.logger.info("generating symbol")
        net = self.mod.sym_gen(cfg.hyper_params)
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
            self.toolbox_init()

    @staticmethod
    def config(cfg=None, **kwargs):
        cfg = Configuration(
            **kwargs
        ) if cfg is None else cfg
        if not isinstance(cfg, Configuration):
            cfg = Configuration.load(cfg)
        cfg.dump(override=True)
        return cfg

    @staticmethod
    def get_module(cfg):
        # todo 到Parameters定义处定义相关参数
        mod = Module(cfg)
        mod.logger.info(str(mod))
        filename = mod.dump_configuration()
        mod.logger.info("parameters saved to %s" % filename)
        return mod

    def viz(self):
        mod = self.mod
        net = self.net

        # optional 3 可视化检查网络
        mod.logger.info("visualizing symbol")
        net_viz(net, mod.cfg)

    def set_loss(self, bp_loss_f=None, loss_function=None):
        bp_loss_f = {

        } if bp_loss_f is None else bp_loss_f

        assert bp_loss_f is not None and len(bp_loss_f) == 1

        loss_function = {

        } if loss_function is None else loss_function
        loss_function.update(bp_loss_f)

        self.bp_loss_f = bp_loss_f
        self.loss_function = loss_function

    def toolbox_init(
            self,
            evaluation_formatter_parameters=None,
            validation_logger_mode="w", informer_silent=False,
    ):

        from longling.lib.clock import Clock
        from longling.lib.utilog import config_logging
        from longling.ML.toolkit.formatter import EvalFormatter
        from longling.ML.toolkit.monitor import MovingLoss, \
            ConsoleProgressMonitor

        self.toolbox = {
            "monitor": dict(),
            "timer": None,
            "formatter": dict(),
        }

        mod = self.mod
        params = self.mod.cfg

        # 4.1 todo 定义损失函数
        # bp_loss_f 定义了用来进行 back propagation 的损失函数，
        # 有且只能有一个，命名中不能为 *_\d+ 型

        assert self.loss_function is not None

        loss_monitor = MovingLoss(self.loss_function)

        # 4.1 todo 初始化一些训练过程中的交互信息
        timer = Clock()

        console_progress_monitor = ConsoleProgressMonitor(
            loss_index=[name for name in self.loss_function],
            end_epoch=params.end_epoch - 1,
            silent=informer_silent
        )

        validation_logger = config_logging(
            filename=params.model_dir + "result.log",
            logger="%s-validation" % params.model_name,
            mode=validation_logger_mode,
            log_format="%(message)s",
        )

        # set evaluation formatter
        evaluation_formatter_parameters = {} \
            if evaluation_formatter_parameters is None \
            else evaluation_formatter_parameters

        evaluation_formatter = EvalFormatter(
            logger=validation_logger,
            dump_file=mod.cfg.validation_result_file,
            **evaluation_formatter_parameters
        )

        self.toolbox["monitor"]["loss"] = loss_monitor
        self.toolbox["monitor"]["progress"] = console_progress_monitor
        self.toolbox["timer"] = timer
        self.toolbox["formatter"]["evaluation"] = evaluation_formatter

    def load_data(self, params=None):
        mod = self.mod
        params = mod.cfg if params is None else params

        # 4.2 todo 定义数据加载
        mod.logger.info("loading data")
        data = get_data_iter(params=params)

        return data

    def model_init(
            self,
            load_epoch=None, force_init=False, params=None,
            allow_reinit=True, trainer=None,
            **kwargs
    ):
        mod = self.mod
        net = self.net
        params = mod.cfg if params is None else params
        begin_epoch = params.begin_epoch

        if self.initialized and not force_init:
            mod.logger.warning("model has been initialized, skip model_init")

        load_epoch = load_epoch if load_epoch is not None else begin_epoch

        # 5 todo 初始化模型
        model_file = kwargs.get(
            mod.epoch_params_filename(load_epoch), "init_model_file"
        )
        try:
            net = mod.load(net, load_epoch, params.ctx)
            mod.logger.info(
                "load params from existing model file "
                "%s" % model_file
            )
        except FileExistsError:
            if allow_reinit:
                mod.logger.info("model doesn't exist, initializing")
                Module.net_initialize(net, params.ctx)
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
            net, optimizer=params.optimizer,
            optimizer_params=params.optimizer_params,
            select=params.train_select
        ) if trainer is None else trainer

    def train_net(self, train_data, eval_data, trainer=None):
        mod = self.mod
        params = self.mod.cfg
        net = self.net

        bp_loss_f = self.bp_loss_f
        loss_function = self.loss_function
        toolbox = self.toolbox

        batch_size = mod.cfg.batch_size
        begin_epoch = mod.cfg.begin_epoch
        end_epoch = mod.cfg.end_epoch
        ctx = mod.cfg.ctx

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
            ctx=ctx,
            toolbox=toolbox,
            prefix=mod.prefix,
            save_epoch=params.save_epoch,
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

        batch_size = mod.cfg.batch_size
        ctx = mod.cfg.ctx

        trainer = self.trainer if trainer is None else trainer
        mod.fit_f(
            net=net, batch_size=batch_size, batch_data=batch_data,
            trainer=trainer,
            bp_loss_f=bp_loss_f, loss_function=loss_function,
            loss_monitor=loss_monitor,
            ctx=ctx,
        )

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def call(self, x, ctx=None):
        # call forward for single data

        # optional
        # pre process x

        # convert the data to ndarray
        ctx = self.mod.cfg.ctx if ctx is None else ctx
        x = mx.nd.array([x], dtype='float32', ctx=ctx)

        # forward
        outputs = self.net(x).asnumpy().tolist()[0]

        return outputs

    def batch_call(self, x):
        # call forward for batch data
        # notice: do not use too big batch size

        # optional
        # pre process x

        # convert the data to ndarray
        x = mx.nd.array(x, dtype='float32', ctx=self.mod.cfg.ctx)

        # forward
        outputs = self.net(x).asnumpy().tolist()

        return outputs

    def transform(self, data):
        return transform(data, self.mod.cfg)

    @staticmethod
    def train(reinforcement=False, **kwargs):
        module = ModelName(**kwargs)
        module.set_loss()
        # module.viz()

        if not reinforcement:
            module.toolbox_init()
            module.model_init(**kwargs)
        else:
            # 增量学习，从某一轮或某个参数配置继续训练
            assert "init_model_file" in kwargs or "load_epoch" in kwargs
            module = ModelName(**kwargs)
            module.toolbox_init(validation_logger_mode="a")

        train_data = module.load_data()
        valid_data = module.load_data()
        module.train_net(train_data, valid_data)

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
    def test(test_epoch, dump_file=None, **kwargs):
        from longling.ML.toolkit.formatter import EvalFormatter
        formatter = EvalFormatter(dump_file=dump_file)
        module = ModelName.load(test_epoch, **kwargs)

        test_data = module.load_data()
        eval_result = module.mod.eval(module.net, test_data)
        formatter(
            tips="test",
            eval_name_value=eval_result
        )
        return eval_result

    @staticmethod
    def run(default_entry="train"):
        cfg_parser = ConfigurationParser(Configuration)
        cfg_parser.add_subcommand(cfg_parser.func_spec(ModelName.config))
        cfg_parser.add_subcommand(cfg_parser.func_spec(ModelName.train))
        cfg_parser.add_subcommand(cfg_parser.func_spec(ModelName.test))
        cfg_parser.add_subcommand(cfg_parser.func_spec(ModelName.load))
        cfg_kwargs = cfg_parser(["config"])

        if "subcommand" in cfg_kwargs:
            subcommand = cfg_kwargs["subcommand"]
            del cfg_kwargs["subcommand"]
        else:
            subcommand = default_entry
        eval("ModelName.%s" % subcommand)(**cfg_kwargs)


if __name__ == '__main__':
    ModelName.run()
