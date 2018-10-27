# coding: utf-8
# Copyright @tongshiwei

try:
    from .GluonModule import *
except (SystemError, ModuleNotFoundError):
    from GluonModule import *


def train_module_name(**kwargs):
    module = module_name(**kwargs)

    module.viz()

    module.package_init()

    train_data, test_data = module.load_data()

    module.model_init(**kwargs)
    module.train(train_data, test_data)


class module_name(object):
    def __init__(self, load_epoch=None, **kwargs):
        # 1 配置参数初始化
        # todo 到Parameters定义处定义相关参数
        params = Parameters(
            **kwargs
        )

        mod = GluonModule(params)
        mod.logger.info(str(mod))

        self.mod = mod

        # 2 todo 定义网络结构
        # 2.1 重新生成
        mod.logger.info("generating symbol")
        net = mod.sym_gen()
        # 2.2 装载已有模型, export 出来的文件
        # net = mod.load(begin_epoch)
        # net = GluonModule.load_net(filename)

        self.net = net

        if load_epoch is not None:
            self.model_init(load_epoch)

        self.bp_loss_f = None
        self.loss_function = None
        self.losses_monitor = None
        self.informer = None
        self.timer = None
        self.evaluator = None

        self.initialized = False

    def viz(self):
        mod = self.mod

        # optional 3 可视化检查网络
        mod.logger.info("visualizing symbol")
        # net_viz(net, mod.params)

    def package_init(self):
        from longling.lib.clock import Clock
        from longling.lib.utilog import config_logging
        from longling.framework.ML.MXnet.mx_gluon.gluon_toolkit import TrainBatchInformer, Evaluator, MovingLosses
        from longling.framework.ML.MXnet.mx_gluon.gluon_sym import PairwiseLoss, SoftmaxCrossEntropyLoss

        mod = self.mod
        params = self.mod.params

        # 4.1 todo 定义损失函数
        # bp_loss_f 定义了用来进行 back propagation 的损失函数，命名中不能出现 下划线
        bp_loss_f = {
            "pairwise-loss": PairwiseLoss(None, -1, margin=1),
            "cross-entropy": gluon.loss.SoftmaxCrossEntropyLoss(),
        }
        loss_function = {

        }
        loss_function.update(bp_loss_f)
        losses_monitor = MovingLosses(loss_function)

        # 4.1 todo 初始化一些训练过程中的交互信息
        timer = Clock()
        informer = TrainBatchInformer(loss_index=[name for name in loss_function], end_epoch=params.end_epoch - 1)
        validation_logger = config_logging(
            filename=params.model_dir + "result.log",
            logger="%s-validation" % params.model_name,
            mode="w",
            log_format="%(message)s",
        )
        from longling.framework.ML.MXnet.metric import PRF, Accuracy
        evaluator = Evaluator(
            # metrics=[PRF(argmax=False), Accuracy(argmax=False)],
            # model_ctx=ctx,
            logger=validation_logger,
            log_f=mod.params.validation_result_file
        )

        self.bp_loss_f = bp_loss_f
        self.loss_function = loss_function
        self.losses_monitor = losses_monitor
        self.informer = informer
        self.timer = timer
        self.evaluator = evaluator

    def load_data(self):
        mod = self.mod

        # 4.2 todo 定义数据加载
        mod.logger.info("loading data")
        train_data = GluonModule.get_data_iter()
        test_data = GluonModule.get_data_iter()

        return train_data, test_data

    def model_init(self, load_epoch=None, force_init=False):
        mod = self.mod

        if self.initialized and not force_init:
            mod.logger.warning("model has been initialized, skip model_init")

        load_epoch = load_epoch if load_epoch is not None else self.mod.params.begin_epoch

        # 5 todo 初始化模型
        # try:
        #     net = mod.load(net, begin_epoch, params.ctx)
        #     mod.logger.info("load params from existing model file %s" % mod.prefix + "-%04d.parmas" % begin_epoch)
        # except FileExistsError:
        #     mod.logger.info("model doesn't exist, initializing")
        #     module_nameModule.net_initialize(net, ctx)
        # self.initialized = True

        # # optional
        # # todo whether to use static symbol to accelerate, do not invoke this method for dynamic structure like rnn
        # net.hybridize()

    def train(self, train_data, test_data):
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

        assert bp_loss_f and loss_function and losses_monitor and informer and timer and evaluator, \
            "make sure these variable have been initialized, " \
            "check init method and make sure init method has been called"

        # 6 todo 训练
        trainer = GluonModule.get_trainer(net, optimizer=params.optimizer, optimizer_params=params.optimizer_params)
        mod.logger.info("start training")
        mod.fit(
            net=net, begin_epoch=begin_epoch, end_epoch=end_epoch, batch_size=batch_size,
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

    def __call__(self, *args, **kwargs):
        pass

    def batch_call(self):
        pass


if __name__ == '__main__':
    train_module_name()
