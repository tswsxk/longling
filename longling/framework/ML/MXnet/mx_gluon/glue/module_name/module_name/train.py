# coding: utf-8
# Copyright @tongshiwei

from longling.lib.clock import Clock
from longling.lib.utilog import config_logging
from longling.framework.ML.MXnet.mx_gluon.gluon_toolkit import TrainBatchInformer, Evaluator, MovingLosses
from longling.framework.ML.MXnet.viz import plot_network, VizError
from longling.framework.ML.MXnet.mx_gluon.gluon_sym import PairwiseLoss, SoftmaxCrossEntropyLoss

try:
    from .GluonModule import *
except ModuleNotFoundError:
    from GluonModule import *


def train_module_name(**kwargs):
    # 1 配置参数初始化
    # todo 到Parameters定义处定义相关参数
    params = Parameters(
        **kwargs
    )

    mod = GluonModule(params)
    mod.logger.info(str(mod))

    batch_size = mod.params.batch_size
    begin_epoch = mod.params.begin_epoch
    end_epoch = mod.params.end_epoch
    ctx = mod.params.ctx

    # 2 todo 定义网络结构并保存
    # 2.1 重新生成
    # mod.logger.info("generating symbol")
    # net = mod.sym_gen()
    # 2.2 装载已有模型
    # net = mod.load(epoch)
    # net = GluonModule.load_net(filename)

    # 3 可视化检查网络
    # net_viz(net, mod.params)

    # 5 todo 定义损失函数
    # bp_loss_f 定义了用来进行 back propagation 的损失函数，命名中不能出现 下划线
    # bp_loss_f = {
    #     "pairwise_loss": PairwiseLoss(None, -1, margin=1),
    #     "cross-entropy": gluon.loss.SoftmaxCrossEntropyLoss(),
    # }
    # loss_function = {
    #
    # }
    # loss_function.update(bp_loss_f)
    # losses_monitor = MovingLosses(loss_function)

    # 5 todo 初始化一些训练过程中的交互信息
    # timer = Clock()
    # informer = TrainBatchInformer(loss_index=[name for name in loss_function], end_epoch=params.end_epoch - 1)
    # validation_logger = config_logging(
    #     filename=params.model_dir + "result.log",
    #     logger="%s-validation" % params.model_name,
    #     mode="w",
    #     log_format="%(message)s",
    # )
    # from longling.framework.ML.MXnet.metric import PRF, Accuracy
    # evaluator = Evaluator(
    #     # metrics=[PRF(argmax=False), Accuracy(argmax=False)],
    #     model_ctx=ctx,
    #     logger=validation_logger,
    #     log_f=mod.params.validation_result_file
    # )

    # 4 todo 定义数据加载
    # mod.logger.info("loading data")
    # train_data = GluonModule.get_data_iter()
    # test_data = GluonModule.get_data_iter()

    # 6 todo 训练
    # 直接装载已有模型，确认这一步可以执行的话可以忽略 2 3 4
    # mod.logger.info("start training")
    # try:
    #     net = mod.load(net, begin_epoch, params.ctx)
    #     mod.logger.info("load params from existing model file %s" % mod.prefix + "-%04d.parmas" % begin_epoch)
    # except FileExistsError:
    #     mod.logger.info("model doesn't exist, initializing")
    #     module_nameModule.net_initialize(net, ctx)
    # trainer = GluonModule.get_trainer(net, optimizer=params.optimizer, optimizer_params=params.optimizer_params)
    # # net.hybridize()  # todo whether to use static symbol to accelerate
    # mod.logger.info("start training")
    # mod.fit(
    #     net=net, begin_epoch=begin_epoch, end_epoch=end_epoch, batch_size=batch_size,
    #     train_data=train_data,
    #     trainer=trainer, bp_loss_f=bp_loss_f,
    #     loss_function=loss_function, losses_monitor=losses_monitor,
    #     test_data=test_data,
    #     ctx=ctx,
    #     informer=informer, epoch_timer=timer, evaluator=evaluator,
    #     prefix=mod.prefix,
    # )
    # net.export(mod.prefix)

    # optional todo 评估
    # GluonModule.eval()
