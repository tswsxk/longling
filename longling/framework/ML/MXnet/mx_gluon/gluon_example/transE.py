# coding:utf-8
# created by tongshiwei on 2018/7/13
"""
对gluon的训练、测试过程进行封装
fit_f = epoch_loop(batch_loop(fit_f))
复制此文件以进行修改
大量使用 staticmethod 并用 get_params 对参数进行分离的原因是因为耦合性太高会导致改起来不太方便
可能修改的地方用 todo 标出
"""
import json
import os

from collections import OrderedDict

from tqdm import tqdm

import mxnet as mx
from mxnet import autograd, nd
from mxnet import gluon

from longling.lib.utilog import config_logging, LogLevel
from longling.lib.clock import Clock
from longling.framework.ML.MXnet.mx_gluon.gluon_toolkit import TrainBatchInformer, Evaluator


########################################################################
# write the user function here

#########################################################################


# todo 重命名eval_transE函数到需要的模块名
def eval_transE():
    pass


# todo 重命名use_transE函数到需要的模块名
def use_transE():
    pass


# todo 重命名train_transE函数到需要的模块名
def train_transE():
    # 1 配置参数初始化
    root = "../../../../"
    model_name = "transE"
    model_dir = root + "data/gluon/%s/" % model_name

    mod = TransEModule(
        model_dir=model_dir,
        model_name=model_name,
        ctx=mx.cpu()
    )
    logger = config_logging(logger=model_name, console_log_level=LogLevel.INFO)
    logger.info(str(mod))

    # 2 todo 定义网络结构并保存
    # 2.1 重新生成
    # logger.info("generating symbol")
    # net = TransEModule.sym_gen()
    # net.hybridize()
    # net.export(model_dir + model_name)
    # 2.2 装载已有模型
    # net = mod.load(epoch)
    # net = TransEModule.load_net(filename)

    # 3 todo 自行设定网络输入，可视化检查网络，检查后删除
    # logger.info("visualization")
    # viz_shape = {'data': (batch_size,) + (1, )}
    # x = mx.sym.var("data")
    # sym = net(x)
    # plot_network(
    #     nn_symbol=sym,
    #     save_path=model_dir + "plot/network",
    #     shape=viz_shape,
    #     node_attrs={"fixedsize": "false"},
    #     view=False
    # )

    # 5 todo 定义训练相关参数
    # logger.info("visualization")
    # begin_epoch = 0
    # epoch_num = 200

    # 5 todo 定义损失函数
    # bp_loss_f 定义了用来进行 back propagation 的损失函数
    # bp_loss_f = {"pairwise_loss": PairwiseLoss(None, -1, margin=1)}
    # loss_function = {
    #
    # }
    # loss_function.update(bp_loss_f)

    # 5 todo 初始化一些训练过程中的交互信息
    # timer = Clock()
    # informer = TrainBatchInformer(loss_index=[name for name in loss_function], epoch_num=epoch_num - 1)
    # validation_logger = config_logging(
    #     filename=model_dir + "result.log",
    #     logger="%s-validation" % model_name,
    #     mode="w",
    #     log_format="%(message)s",
    # )
    # evaluator = Evaluator(
    #     # metrics=eval_metrics,
    #     model_ctx=model_ctx,
    #     logger=validation_logger,
    #     log_f=validation_result_file
    # )

    # 4 todo 定义数据加载
    # train_data = TransEModule.get_data_iter()
    # test_data = TransEModule.get_data_iter()

    # 4 todo 加载数据后, 可视化检查网络
    # viz_shape = {'data': (batch_size,) + data_shape}
    # x = mx.sym.var("data")
    # sym = net(x)
    # plot_network(
    #     nn_symbol=sym,
    #     save_path=model_dir + "plot/network",
    #     shape=viz_shape,
    #     node_attrs={"fixedsize": "false"},
    #     view=False
    # )

    # 6 todo 训练
    # 直接装载已有模型，确认这一步可以执行的话可以忽略 2 3 4
    # try:
    #     net = mod.load(begin_epoch)
    # except FileExistsError:
    #     TransEModule.net_initialize()
    # trainer = TransEModule.get_trainer()
    # func = TransEModule.fit()

    # optional todo 评估
    # TransEModule.eval()

    # 7 todo 关闭输入输出流
    # evaluator.close()


class TransEModule(object):
    """
    模块模板
    train 修改流程

    # 1
    修改 __init__ 和 params 方法
    TransEModule(....) 初始化一些通用的参数，比如模型的存储路径等

    # 2
    定义网络部分，命名为
    sym_gen(), 可以有多个
    也可以使用直接装载的方式生成已有网络

    # 3
    使用可视化方法进行检查网络是否搭建得没有问题
    可视化方法命名为 visualization

    # 4
    定义数据装载部分，可以有多个，命名为
    get_{train/test/data}_iter

    # 5
    定义训练相关的参数及交互提示信息

    # 6
    定义 net_initialize 部分
    定义训练部分

    # 7
    关闭输入输出流

    # optional
    定义 eval 评估模型方法
    定义 load 加载模型方法
    """

    def __init__(self, model_dir, model_name, ctx=mx.cpu()):
        """

        Parameters
        ----------
        model_dir: str
            The directory to store the model and the corresponding files such as log
        model_name: str
            The name of this model
        ctx: Context or list of Context
            Defaults to ``mx.cpu()``.
        """
        # 初始化一些通用的参数
        self.model_dir = os.path.abspath(model_dir)
        self.model_name = model_name
        self.validation_result_file = os.path.abspath(model_dir + "result")
        self.prefix = os.path.join(self.model_dir, self.model_name)
        self.ctx = ctx

    def __str__(self):
        """
        显示模块参数
        Display the necessary params of this Module
        Returns
        -------

        """
        string = ["Params"]
        for k, v in self.params.items():
            string.append("%s: %s" % (k, v))
        return "\n".join(string)

    @property
    def params(self):
        # 此函数用来将需要用到的参数以字典形式返回
        params = OrderedDict()
        params['model_name'] = self.model_name
        params['ctx'] = self.ctx
        params['prefix'] = self.prefix
        params['model_dir'] = self.model_dir
        params['validation_result_file'] = self.validation_result_file
        return params

    @staticmethod
    def load_net(filename):
        """
        Load the existing net parameters
        Parameters
        ----------
        filename: str
            The model file
        Returns
        -------
        The initialized net
        """
        # 根据文件名装载已有的网络参数
        if not os.path.isfile(filename):
            raise FileExistsError
        model = nd.load(filename)
        net = gluon.nn.HybridSequential()
        with net.name_scope():
            net.add(model)
        return net

    def load(self, epoch):
        """"
        Load the existing net parameters
        Parameters
        ----------
        epoch: str or int
            The epoch which specify the model
        Returns
        -------
        The initialized net
        """
        # 根据起始轮次装载已有的网络参数
        filename = self.prefix + "-%04d.parmas" % epoch
        return self.load_net(filename)

    @staticmethod
    def get_train_iter(filename, batch_size, entities_map=None, relations_map=None):
        pos_subs, pos_rels, pos_objs = [], [], []
        neg_subs, neg_rels, neg_objs = [], [], []
        with open(filename) as f:
            for line in tqdm(f, desc="reading file[%s]" % filename):
                if not line.strip():
                    continue
                pos_neg = json.loads(line)
                pos_sub, pos_rel, pos_obj = pos_neg["x"]
                pos_subs.append(entities_map[pos_sub])
                pos_rels.append(relations_map[pos_rel])
                pos_objs.append(entities_map[pos_obj])
                neg_sub, neg_rel, neg_obj = pos_neg["z"]
                neg_subs.append(entities_map[neg_sub])
                neg_rels.append(relations_map[neg_rel])
                neg_objs.append(entities_map[neg_obj])

        return gluon.data.DataLoader(
            gluon.data.ArrayDataset(pos_subs, pos_rels, pos_objs, neg_subs, neg_rels, neg_objs), batch_size=batch_size,
            shuffle=True)

    @staticmethod
    def get_test_iter(filename, entities_map=None, relations_map=None):
        with open(filename) as f:
            for i, line in f:
                if not line.strip():
                    continue
                pos_neg = json.loads(line)
                pos_sub, pos_rel, pos_obj = pos_neg["x"]
                if entities_map:
                    pos_sub = entities_map[pos_sub]
                    pos_obj = entities_map[pos_obj]
                if relations_map:
                    pos_rel = relations_map[pos_rel]
                negs = []
                for neg_triple in pos_neg["z"]:
                    neg_sub, neg_rel, neg_obj = neg_triple
                    if entities_map:
                        neg_sub = entities_map[neg_sub]
                        neg_rel = relations_map[neg_rel]
                        neg_obj = entities_map[neg_obj]
                    negs.append((neg_sub, neg_rel, neg_obj))
                yield [(pos_sub, pos_rel, pos_obj)]

    @staticmethod
    def eval():
        # 在这里定义数据评估方法
        return

    @staticmethod
    def sym_gen():
        # 在这里定义网络结构
        return

    # 以下部分定义训练相关的方法
    @staticmethod
    def net_initialize(net, model_ctx, initializer=mx.init.Normal(sigma=.1)):
        # 初始化网络参数
        net.collect_params().initialize(initializer, ctx=model_ctx)

    @staticmethod
    def get_trainer(net, optimizer='sgd', optimizer_params={'learning_rate': .01}):
        # 把优化器安装到网络上
        trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
        return trainer

    def fit(
            self,
            net, begin_epoch, epoch_num,
            train_data,
            trainer, bp_loss_f,
            loss_function, losses_monitor=None,
            test_data=None,
            ctx=mx.cpu(),
            informer=None, epoch_timer=None, evaluator=None,
            **kwargs
    ):
        """
        API for train
        Parameters
        ----------
        net: HybridBlock
            The network which has been initialized or loaded from the existed model
        begin_epoch:
            The begin epoch of this train procession
        epoch_num:
            The end epoch of this train procession
        train_data: Iterable
            The data used for this train procession, NOTICE: should have been divided to batches
        trainer:
            The trainer used to update the parameters of the net
        bp_loss_f: dict with only one value and one key
            The function to compute the loss for the procession of back propagation
        loss_function: dict of function
            Some other measurement in addition to bp_loss_f
        losses_monitor: LossesMonitor
            Default to ``None``
        test_data: Iterable
            The data used for the evaluation at the end of each epoch, NOTICE: should have been divided to batches
            Default to ``None``
        ctx: Context or list of Context
            Defaults to ``mx.cpu()``.
        informer: TrainBatchInformer
            Default to ``None``
        epoch_timer: Clock
            Default to ``None``
        evaluator: Evaluator
            Default to ``None``
        kwargs
        Returns
        -------
        """
        # 此方法可以直接使用
        return self.epoch_loop(self.batch_loop(self._fit_f))(
            net=net, begin_epoch=begin_epoch, epoch_num=epoch_num,
            train_data=train_data,
            trainer=trainer, bp_loss_f=bp_loss_f,
            loss_function=loss_function, losses_monitor=losses_monitor,
            test_data=test_data,
            ctx=ctx,
            batch_informer=informer, epoch_timer=epoch_timer, evaluator=evaluator,
            **kwargs
        )

    @staticmethod
    def epoch_loop(batch_loop):
        """
        此函数包裹批次训练过程，形成轮次训练过程
        只需要修改 decorator 部分就可以
        Parameters
        ----------
        batch_loop: Function
            The function defining how the batch training processes
        Returns
        -------
            Decorator
        """

        def decorator(
                net, begin_epoch, epoch_num,
                train_data,
                trainer, bp_loss_f,
                loss_function, losses_monitor=None,
                test_data=None,
                ctx=mx.cpu(),
                informer=None, epoch_timer=None, evaluator=None,
                **kwargs
        ):
            """
            The true body of the epoch loop
            Parameters
            ----------
            net: HybridBlock
                The network which has been initialized or loaded from the existed model
            begin_epoch:
                The begin epoch of this train procession
            epoch_num:
                The end epoch of this train procession
            train_data: Iterable
                The data used for this train procession, NOTICE: should have been divided to batches
            trainer:
                The trainer used to update the parameters of the net
            bp_loss_f: dict with only one value and one key
                The function to compute the loss for the procession of back propagation
            loss_function: dict of function
                Some other measurement in addition to bp_loss_f
            losses_monitor: LossesMonitor
                Default to ``None``
            test_data: Iterable
                The data used for the evaluation at the end of each epoch, NOTICE: should have been divided to batches
                Default to ``None``
            ctx: Context or list of Context
                Defaults to ``mx.cpu()``.
            informer: TrainBatchInformer
                Default to ``None``
            epoch_timer: Clock
                Default to ``None``
            evaluator: Evaluator
                Default to ``None``
            kwargs
            Returns
            -------

            """
            # 参数修改时需要同步修改 fit 函数中的参数
            # 定义轮次训练过程
            for epoch in range(begin_epoch, epoch_num):
                # initial
                if losses_monitor:
                    losses_monitor.reset()
                if informer:
                    informer.batch_start(epoch)
                if epoch_timer:
                    epoch_timer.start()
                batch_num, loss_values = batch_loop(
                    net=net,
                    train_data=train_data,
                    trainer=trainer, bp_loss_f=bp_loss_f,
                    loss_function=loss_function, losses_monitor=losses_monitor,
                    ctx=ctx,
                    batch_informer=informer,
                )
                if informer:
                    informer.batch_end(batch_num)

                train_time = epoch_timer.end(wall=True) if epoch_timer else None

                # todo 定义每一轮结束后的模型评估方法
                # test_eval_res = evaluator.evaluate(test_data, net, stage="test")
                # print(evaluator.format_eval_res(epoch, test_eval_res, loss_values, train_time,
                #                                 logger=evaluator.logger, log_f=evaluator.log_f)[0])

                # todo 定义模型保存方案
                # if kwargs.get('prefix'):
                #     net.save_params(kwargs['prefix'] + "-%04d.parmas" % (epoch + 1))

        return decorator

    @staticmethod
    def batch_loop(fit_f):
        """
        此函数包裹训练过程，形成批次训练过程
        只需要修改 decorator 部分就可以
        Parameters
        ----------
        fit_f

        Returns
        -------
        decorator 装饰器
        """

        def decorator(
                net,
                train_data,
                trainer, bp_loss_f,
                loss_function, losses_monitor=None,
                ctx=mx.cpu(),
                batch_informer=None,
        ):
            """
            The true body of batch loop
            Parameters
            ----------
            net: HybridBlock
                The network which has been initialized or loaded from the existed model
            train_data: Iterable
                The data used for this train procession, NOTICE: should have been divided to batches
            trainer:
                The trainer used to update the parameters of the net
            bp_loss_f: dict with only one value and one key
                The function to compute the loss for the procession of back propagation
            loss_function: dict of function
                Some other measurement in addition to bp_loss_f
            losses_monitor: LossesMonitor
                Default to ``None``
            ctx: Context or list of Context
                Defaults to ``mx.cpu()``.
            batch_informer: TrainBatchInformer
                Default to ``None``

            Returns
            -------
            loss_values: dict
                Recorded the loss values computed by the loss_function
            i: int
                The total batch num of this epoch
            """
            # 定义批次训练过程
            # 这部分改动可能会比较多，主要是train_data的输出部分
            # write batch loop body here
            for i, (data, label) in enumerate(train_data):
                fit_f(
                    net=net,
                    data=data, label=label,
                    trainer=trainer, bp_loss_f=bp_loss_f, loss_function=loss_function,
                    losses_monitor=losses_monitor,
                    ctx=ctx,
                )
                if batch_informer:
                    loss_values = [loss for loss in losses_monitor.values()]
                    batch_informer.batch_report(i, loss_value=loss_values)
            loss_values = {name: loss for name, loss in losses_monitor.items()}.items()
            return loss_values, i

        return decorator

    @staticmethod
    def _fit_f(net,
               data, label,
               trainer, bp_loss_f, loss_function, losses_monitor=None,
               ctx=mx.cpu()
               ):
        """
        Defined how each step of batch train goes
        Parameters
        ----------
        net: HybridBlock
            The network which has been initialized or loaded from the existed model
        data: Iterable
            The data for train
        label: Iterable
            The lable for train
        trainer:
            The trainer used to update the parameters of the net
        bp_loss_f: dict with only one value and one key
            The function to compute the loss for the procession of back propagation
        loss_function: dict of function
            Some other measurement in addition to bp_loss_f
        losses_monitor: LossesMonitor
            Default to ``None``
        ctx: Context or list of Context
            Defaults to ``mx.cpu()``.

        Returns
        -------

        """
        # 此函数定义训练过程

        # todo
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)

        bp_loss = None
        with autograd.record():
            output = net(data)  # todo
            for name, func in loss_function.items():
                loss = func(output, label)  # todo
                if name in bp_loss_f:
                    bp_loss = loss
                loss_value = nd.mean(loss).asscalar()
                if losses_monitor:
                    losses_monitor.update(name, loss_value)

        assert bp_loss is not None
        bp_loss.backward()
        trainer.step(data.shape[0])


if __name__ == '__main__':
    train_transE()
