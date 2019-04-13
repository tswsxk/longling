# coding: utf-8
# Copyright @tongshiwei
import mxnet as mx

try:
    from .Module import *
except (SystemError, ModuleNotFoundError):
    from Module import *


class ModelName(object):
    def __init__(
            self, load_epoch=None, params=None, package_init=False, **kwargs
    ):
        # 1 配置参数初始化
        # todo 到Parameters定义处定义相关参数
        params = Parameters(
            **kwargs
        ) if params is None else params

        mod = Module(params)
        mod.logger.info(str(mod))

        filename = mod.dump_parameters()
        mod.logger.info("parameters saved to %s" % filename)

        self.mod = mod

        # 2 todo 定义网络结构
        # 2.1 重新生成
        mod.logger.info("generating symbol")
        net = mod.sym_gen()
        # 2.2 装载已有模型, export 出来的文件
        # net = mod.load(begin_epoch)
        # net = GluonModule.load_net(filename)

        self.net = net
        self.initialized = False

        if load_epoch is not None:
            self.model_init(load_epoch)

        self.bp_loss_f = None
        self.loss_function = None
        self.losses_monitor = None
        self.informer = None
        self.timer = None
        self.evaluator = None
        self.trainer = None

        if package_init:
            self.package_init()

    def viz(self):
        mod = self.mod
        net = self.net

        # optional 3 可视化检查网络
        mod.logger.info("visualizing symbol")
        net_viz(net, mod.params)

    def package_init(self, evaluation_formatter_parameters=None,
                     validation_logger_mode="w", informer_silent=False):

        from longling.lib.clock import Clock
        from longling.lib.utilog import config_logging
        from longling.ML.toolkit.formatter import EvalFormatter
        from longling.ML.toolkit.monitor import MovingLoss, \
            ConsoleProgressMonitor

        mod = self.mod
        params = self.mod.params

        # 4.1 todo 定义损失函数
        # bp_loss_f 定义了用来进行 back propagation 的损失函数，
        # 有且只能有一个，命名中不能为 *_\d+ 型
        bp_loss_f = {

        }

        assert bp_loss_f is not None and len(bp_loss_f) == 1

        loss_function = {

        }
        loss_function.update(bp_loss_f)
        losses_monitor = MovingLoss(loss_function)

        # 4.1 todo 初始化一些训练过程中的交互信息
        timer = Clock()
        informer = ConsoleProgressMonitor(
            loss_index=[name for name in loss_function],
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
            log_f=mod.params.validation_result_file,
            **evaluation_formatter_parameters
        )

        self.bp_loss_f = bp_loss_f
        self.loss_function = loss_function
        self.losses_monitor = losses_monitor
        self.informer = informer
        self.timer = timer
        self.evaluator = evaluation_formatter

    def load_data(self, params=None):
        mod = self.mod
        params = mod.params if params is None else params

        # 4.2 todo 定义数据加载
        mod.logger.info("loading data")
        train_data = Module.get_data_iter(params=params)
        test_data = Module.get_data_iter(params=params)

        return train_data, test_data

    def model_init(self, load_epoch=None, force_init=False, params=None,
                   **kwargs):
        mod = self.mod
        net = self.net
        params = mod.params if params is None else params
        begin_epoch = params.begin_epoch

        if self.initialized and not force_init:
            mod.logger.warning("model has been initialized, skip model_init")

        load_epoch = load_epoch if load_epoch is not None else begin_epoch

        # 5 todo 初始化模型
        # try:
        #     net = mod.load(net, load_epoch, params.ctx)
        #     mod.logger.info("load params from existing model file %s" % mod.prefix + "-%04d.parmas" % load_epoch)
        # except FileExistsError:
        #     mod.logger.info("model doesn't exist, initializing")
        #     module_nameModule.net_initialize(net, params.ctx)
        # self.initialized = True

        # # optional
        # # todo whether to use static symbol to accelerate, do not invoke this method for dynamic structure like rnn
        # # suggest annotate this until your process worked
        # net.hybridize()

        self.trainer = Module.get_trainer(net, optimizer=params.optimizer,
                                          optimizer_params=params.optimizer_params)

    def train(self, train_data, test_data, trainer=None):
        mod = self.mod
        params = self.mod.params
        net = self.net

        bp_loss_f = self.bp_loss_f
        loss_function = self.loss_function
        losses_monitor = self.losses_monitor
        informer = self.informer
        timer = self.timer
        evaluator = self.evaluator

        batch_size = mod.params.batch_size
        begin_epoch = mod.params.begin_epoch
        end_epoch = mod.params.end_epoch
        ctx = mod.params.ctx

        assert all([bp_loss_f, loss_function, losses_monitor, informer, timer,
                    evaluator]), \
            "make sure these variable have been initialized, " \
            "check init method and make sure package_init method has been called"

        # 6 todo 训练
        trainer = self.trainer if trainer is None else trainer
        mod.logger.info("start training")
        mod.fit(
            net=net, begin_epoch=begin_epoch, end_epoch=end_epoch,
            batch_size=batch_size,
            train_data=train_data,
            trainer=trainer, bp_loss_f=bp_loss_f,
            loss_function=loss_function, losses_monitor=losses_monitor,
            test_data=test_data,
            ctx=ctx,
            informer=informer, epoch_timer=timer, evaluator=evaluator,
            prefix=mod.prefix,
            save_epoch=params.save_epoch,
        )

        # optional
        # # need execute the net.hybridize() before export, and execute forward at least one time
        # 需要在这之前调用 hybridize 方法,并至少forward一次
        # # net.export(mod.prefix)

    def step_fit(self, batch_data, trainer=None):
        mod = self.mod
        net = self.net

        bp_loss_f = self.bp_loss_f
        loss_function = self.loss_function
        losses_monitor = self.losses_monitor

        batch_size = mod.params.batch_size
        ctx = mod.params.ctx

        trainer = self.trainer if trainer is None else trainer
        mod.fit_f(
            net=net, batch_size=batch_size, batch_data=batch_data,
            trainer=trainer,
            bp_loss_f=bp_loss_f, loss_function=loss_function,
            losses_monitor=losses_monitor,
            ctx=ctx,
        )

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def call(self, x, ctx=None):
        # call forward for single data

        # optional
        # pre process x

        # convert the data to ndarray
        ctx = self.mod.params.ctx if ctx is None else ctx
        x = mx.nd.array([x], dtype='float32', ctx=ctx)

        # forward
        outputs = self.net(x).asnumpy().tolist()[0]

        raise NotImplementedError

    def batch_call(self, x, ctx=None):
        # call forward for batch data
        # notice: do not use too big batch size

        # optional
        # pre process x

        # convert the data to ndarray
        x = mx.nd.array(x, dtype='float32', ctx=self.mod.params.ctx)

        # forward
        outputs = self.net(x).asnumpy().tolist()

        raise NotImplementedError

    def pre_process(self, data):
        raise NotImplementedError


def train_module_name(**kwargs):
    module = ModelName(**kwargs)

    module.viz()

    module.package_init()

    train_data, test_data = module.load_data()

    module.model_init(**kwargs)
    module.train(train_data, test_data)


if __name__ == '__main__':
    train_module_name()