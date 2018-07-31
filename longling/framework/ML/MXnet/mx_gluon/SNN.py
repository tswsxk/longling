# coding:utf-8
# created by tongshiwei on 2018/7/13
"""
对gluon的训练、测试过程进行封装
fit_f = epoch_loop(batch_loop(fit_f))
复制此文件以进行修改
大量使用 staticmethod 并用 get_params 对参数进行分离的原因是因为耦合性太高会导致改起来不太方便
可能修改的地方用 todo 标出
"""
import os

from collections import OrderedDict

import mxnet as mx
from mxnet import autograd, nd
from mxnet import gluon

from longling.lib.utilog import config_logging, LogLevel
from longling.lib.clock import Clock
from longling.framework.ML.MXnet.mx_gluon.gluon_toolkit import TrainBatchInformer, Evaluator, MovingLosses
from longling.framework.ML.MXnet.viz import plot_network
from longling.framework.ML.MXnet.mx_gluon.gluon_sym import PairwiseLoss, SoftmaxCrossEntropyLoss


#######################################################################################################################
# write network definition here
class DistanceLoss(gluon.HybridBlock):
    def hybrid_forward(self, F, x, y, *args, **kwargs):
        distance = x - y
        return F.MakeLoss(F.sum(distance * distance, axis=-1))


class SNN(gluon.HybridBlock):
    def __init__(self, dim=256, action_num=10, **kwargs):
        super(SNN, self).__init__(**kwargs)

        with self.name_scope():
            self.action_embedding = gluon.nn.Embedding(action_num, dim)
            self.lstm = gluon.rnn.LSTMCell(dim)
            self.loss = DistanceLoss()
        self.action_len = None

    def hybrid_forward(self, F, action_seq, begin_state, *args, **kwargs):
        actions = self.action_embedding(action_seq)
        states, _ = self.lstm.unroll(self.action_len, actions,
                                     begin_state=[begin_state] * 2,
                                     )
        current_state = states[-1]
        return current_state


#######################################################################################################################

#######################################################################################################################
# write the user function here

#######################################################################################################################


# todo 重命名eval_SNN函数到需要的模块名
def eval_SNN():
    pass


# todo 重命名use_SNN函数到需要的模块名
def use_SNN():
    pass


# todo 重命名train_SNN函数到需要的模块名
def train_SNN():
    # 1 配置参数初始化
    root = "../../../../"
    model_name = "SNN"
    model_dir = root + "data/gluon/%s/" % model_name

    mod = SNNModule(
        model_dir=model_dir,
        model_name=model_name,
        ctx=mx.cpu()
    )
    logger = config_logging(logger=model_name, console_log_level=LogLevel.INFO)
    logger.info(str(mod))

    ############################################################################
    # experiment params

    ############################################################################

    # 2 todo 定义网络结构并保存
    # 2.1 重新生成
    logger.info("generating symbol")
    net = SNNModule.sym_gen()
    # 2.2 装载已有模型
    # net = mod.load(epoch)
    # net = SNNModule.load_net(filename)

    # 5 todo 定义训练相关参数
    begin_epoch = 0
    epoch_num = 10
    batch_size = 128
    ctx = mod.ctx

    # 3 todo 自行设定网络输入，可视化检查网络
    logger.info("visualization")
    from copy import deepcopy
    viz_net = deepcopy(net)
    action_len = 1
    viz_shape = {
        'data': (batch_size,) + (action_len,),
        'state': (batch_size,) + (256,),
    }
    x = mx.sym.var("data")
    state = mx.sym.var("state")
    viz_net.action_len = action_len
    sym = viz_net(x, state)
    viz_net.hybridize()
    plot_network(
        nn_symbol=sym,
        save_path=model_dir + "plot/network",
        shape=viz_shape,
        node_attrs={"fixedsize": "false"},
        view=False
    )

    # 5 todo 定义损失函数
    # bp_loss_f 定义了用来进行 back propagation 的损失函数
    bp_loss_f = {
        # "pairwise_loss": PairwiseLoss(None, -1, margin=1),
        # "cross-entropy": gluon.loss.SoftmaxCrossEntropyLoss(),
        'distance-loss': DistanceLoss(),
    }
    loss_function = {

    }
    loss_function.update(bp_loss_f)
    losses_monitor = MovingLosses(loss_function)

    # 5 todo 初始化一些训练过程中的交互信息
    timer = Clock()
    informer = TrainBatchInformer(loss_index=[name for name in loss_function], epoch_num=epoch_num - 1)
    validation_logger = config_logging(
        filename=model_dir + "result.log",
        logger="%s-validation" % model_name,
        mode="w",
        log_format="%(message)s",
    )
    # from longling.framework.ML.MXnet.metric import PRF, Accuracy
    evaluator = Evaluator(
        # metrics=[PRF(argmax=False), Accuracy(argmax=False)],
        model_ctx=mod.ctx,
        logger=validation_logger,
        log_f=mod.validation_result_file
    )

    # 4 todo 定义数据加载
    logger.info("loading data")
    train_data = SNNModule.get_data_iter()
    test_data = SNNModule.get_data_iter()

    # 6 todo 训练
    # 直接装载已有模型，确认这一步可以执行的话可以忽略 2 3 4
    logger.info("start training")
    try:
        net = mod.load(net, begin_epoch, mod.ctx)
        logger.info("load params from existing model file %s" % mod.prefix + "-%04d.parmas" % begin_epoch)
    except FileExistsError:
        logger.info("model doesn't exist, initializing")
        SNNModule.net_initialize(net, ctx)
    trainer = SNNModule.get_trainer(net)
    mod.fit(
        net=net, begin_epoch=begin_epoch, epoch_num=epoch_num, batch_size=batch_size,
        train_data=train_data,
        trainer=trainer, bp_loss_f=bp_loss_f,
        loss_function=loss_function, losses_monitor=losses_monitor,
        test_data=test_data,
        ctx=ctx,
        informer=informer, epoch_timer=timer, evaluator=evaluator,
        prefix=mod.prefix,
    )
    # net.export(mod.prefix)

    # optional todo 评估
    # SNNModule.eval()

    # 7 todo 关闭输入输出流
    evaluator.close()


class SNNModule(object):
    """
    模块模板
    train 修改流程

    # 1
    修改 __init__ 和 params 方法
    SNNModule(....) 初始化一些通用的参数，比如模型的存储路径等

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
        self.validation_result_file = os.path.abspath(model_dir + "result.json")
        self.prefix = os.path.join(self.model_dir, self.model_name)
        self.ctx = ctx

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

    @staticmethod
    def load_net(filename, net, ctx=mx.cpu()):
        """
        Load the existing net parameters
        Parameters
        ----------
        filename: str
            The model file
        net: HybridBlock
            The network which has been initialized or loaded from the existed model
        ctx: Context or list of Context
                Defaults to ``mx.cpu()``.
        Returns
        -------
        The initialized net
        """
        # 根据文件名装载已有的网络参数
        if not os.path.isfile(filename):
            raise FileExistsError
        return net.load_params(filename, ctx)

    def load(self, net, epoch, ctx=mx.cpu()):
        """"
        Load the existing net parameters
        Parameters
        ----------
        net: HybridBlock
            The network which has been initialized or loaded from the existed model
        epoch: str or int
            The epoch which specify the model
        ctx: Context or list of Context
                Defaults to ``mx.cpu()``.
        Returns
        -------
        The initialized net
        """
        # 根据起始轮次装载已有的网络参数
        filename = self.prefix + "-%04d.parmas" % epoch
        return self.load_net(filename, net, ctx)

    @staticmethod
    def get_data_iter(batch_size=128, num_buckets=10, padding=0):
        # 在这里定义数据加载方法
        # action_seqs = []
        # begin_states = []
        # target_states = []
        import random
        pesudo_num = 1000
        action_seqs = sorted([
            [random.randint(1, 10) for _ in range(random.randint(1, 20))]
            for _ in range(pesudo_num)
        ], key=lambda x: len(x))
        begin_states = [
            [random.random() for _ in range(256)]
            for _ in range(pesudo_num)
        ]
        target_states = [
            [random.random() for _ in range(256)]
            for _ in range(pesudo_num)
        ]
        from gluonnlp.data import FixedBucketSampler, PadSequence
        import numpy as np
        batch_idxes = FixedBucketSampler([len(action_seq) for action_seq in action_seqs], batch_size,
                                         num_buckets=num_buckets)
        batch = []
        for batch_idx in batch_idxes:
            batch_action_seqs = []
            batch_begin_states = []
            batch_target_states = []
            for idx in batch_idx:
                batch_action_seqs.append(action_seqs[idx])
                batch_begin_states.append(begin_states[idx])
                batch_target_states.append(target_states[idx])
            batch_data = []
            padder = PadSequence(max([len(action_seq) for action_seq in batch_action_seqs]), pad_val=padding)
            batch_action_seqs = [padder(action_seq) for action_seq in batch_action_seqs]
            batch_data.append(mx.ndarray.array(np.asarray(batch_action_seqs)))
            batch_data.append(mx.ndarray.array(np.asarray(batch_begin_states)))
            batch_data.append(mx.ndarray.array(np.asarray(batch_target_states)))
            batch.append(batch_data)
        return batch

    @staticmethod
    def sym_gen():
        # 在这里定义网络结构
        return SNN()

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
            net, begin_epoch, epoch_num, batch_size,
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
        begin_epoch: int
            The begin epoch of this train procession
        epoch_num: int
            The end epoch of this train procession
        batch_size: int
                The size of each batch
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
            net=net, begin_epoch=begin_epoch, epoch_num=epoch_num, batch_size=batch_size,
            train_data=train_data,
            trainer=trainer, bp_loss_f=bp_loss_f,
            loss_function=loss_function, losses_monitor=losses_monitor,
            test_data=test_data,
            ctx=ctx,
            informer=informer, epoch_timer=epoch_timer, evaluator=evaluator,
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
                net, begin_epoch, epoch_num, batch_size,
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
            begin_epoch: int
                The begin epoch of this train procession
            epoch_num: int
                The end epoch of this train procession
            batch_size: int
                The size of each batch
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
                loss_values, batch_num = batch_loop(
                    net=net, batch_size=batch_size,
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
                test_eval_res = SNNModule.eval(test_data, net)
                print(evaluator.format_eval_res(epoch, test_eval_res, loss_values, train_time,
                                                logger=evaluator.logger, log_f=evaluator.log_f)[0])

                # todo 定义模型保存方案
                if kwargs.get('prefix'):
                    net.save_params(kwargs['prefix'] + "-%04d.parmas" % (epoch + 1))

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
                net, batch_size,
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
            batch_size: int
                The size of each batch
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
            for i, (action_seq, begin_state, target_state) in enumerate(train_data):
                fit_f(
                    net=net, batch_size=batch_size,
                    action_seq=action_seq, begin_state=begin_state, target_state=target_state,
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
    def eval(test_data, net, eval_func=DistanceLoss()):
        # 在这里定义数据评估方法
        accumulative_loss = 0
        for (action_seq, begin_state, target_state) in test_data:
            net.action_len = len(action_seq[0])
            output = net(action_seq, begin_state)
            accumulative_loss += nd.sum(eval_func(output, target_state)).asscalar()
        return {'accumulative_loss': accumulative_loss}

    @staticmethod
    def _fit_f(net, batch_size,
               action_seq, begin_state, target_state,
               trainer, bp_loss_f, loss_function, losses_monitor=None,
               ctx=mx.cpu()
               ):
        """
        Defined how each step of batch train goes
        Parameters
        ----------
        net: HybridBlock
            The network which has been initialized or loaded from the existed model
        batch_size: int
                The size of each batch
        action_seq: Iterable
            The data for train
        target_state: Iterable
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

        action_seq = action_seq.as_in_context(ctx)
        begin_state = begin_state.as_in_context(ctx)
        target_state = target_state.as_in_context(ctx)

        bp_loss = None
        with autograd.record():
            net.action_len = len(action_seq[0])
            output = net(action_seq, begin_state)
            for name, func in loss_function.items():
                loss = func(output, target_state)
                if name in bp_loss_f:
                    bp_loss = loss
                loss_value = nd.mean(loss).asscalar()
                if losses_monitor:
                    losses_monitor.update(name, loss_value)

        assert bp_loss is not None
        bp_loss.backward()
        trainer.step(batch_size)


if __name__ == '__main__':
    train_SNN()