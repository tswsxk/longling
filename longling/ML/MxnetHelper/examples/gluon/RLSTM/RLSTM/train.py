# coding: utf-8
# create by tongshiwei on 2018/8/5


from longling.lib.clock import Clock
from longling.ML.MxnetHelper.mx_gluon import TrainBatchInformer, Evaluator, MovingLosses
from longling.framework.ML.MXnet.viz import plot_network, VizError
from longling.ML.MxnetHelper.mx_gluon import PairwiseLoss, SoftmaxCrossEntropyLoss

from .RLSTMModule import RLSTMModule, Parameters, net_viz


def train_RLSTM():
    # 1 配置参数初始化
    # todo 到Parameters定义处定义相关参数
    params = Parameters(

    )

    mod = RLSTMModule(params)
    mod.logger.info(str(mod))

    batch_size = mod.params.batch_size
    begin_epoch = mod.params.begin_epoch
    end_epoch = mod.params.end_epoch

    # 2 todo 定义网络结构并保存
    # 2.1 重新生成
    # logger.info("generating symbol")
    # net = mod.sym_gen()
    # 2.2 装载已有模型
    # net = mod.load(epoch)
    # net = RLSTMModule.load_net(filename)

    # 3 可视化检查网络
    # net_viz(net, mod.params, mod.logger)


    # 5 todo 定义损失函数
    # bp_loss_f 定义了用来进行 back propagation 的损失函数
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
    # informer = TrainBatchInformer(loss_index=[name for name in loss_function], epoch_num=epoch_num - 1)
    # validation_logger = config_logging(
    #     filename=model_dir + "result.log",
    #     logger="%s-validation" % model_name,
    #     mode="w",
    #     log_format="%(message)s",
    # )
    # from longling.framework.ML.MXnet.metric import PRF, Accuracy
    # evaluator = Evaluator(
    #     # metrics=[PRF(argmax=False), Accuracy(argmax=False)],
    #     model_ctx=mod.ctx,
    #     logger=validation_logger,
    #     log_f=mod.validation_result_file
    # )

    # 4 todo 定义数据加载
    # logger.info("loading data")
    # train_data = RLSTMModule.get_data_iter()
    # test_data = RLSTMModule.get_data_iter()

    # 6 todo 训练
    # 直接装载已有模型，确认这一步可以执行的话可以忽略 2 3 4
    # logger.info("start training")
    # try:
    #     net = mod.load(net, begin_epoch, mod.ctx)
    #     logger.info("load params from existing model file %s" % mod.prefix + "-%04d.parmas" % begin_epoch)
    # except FileExistsError:
    #     logger.info("model doesn't exist, initializing")
    #     RLSTMModule.net_initialize(net, ctx)
    # RLSTMModule.parameters_stabilize(net)
    # trainer = RLSTMModule.get_trainer(net)
    # net.hybridize() # todo whether to use static symbol to accelerate
    # logger.info("start training")
    # mod.fit(
    #     net=net, begin_epoch=begin_epoch, epoch_num=epoch_num, batch_size=batch_size,
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
    # RLSTMModule.eval()

    # 7 todo 关闭输入输出流
    # evaluator.close()
