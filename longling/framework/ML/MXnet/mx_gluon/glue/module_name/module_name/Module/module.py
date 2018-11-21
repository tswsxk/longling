# coding: utf-8
# Copyright @tongshiwei

from __future__ import absolute_import

import datetime
import os
import shutil

import mxnet as mx
from mxnet import gluon, autograd, nd
from tqdm import tqdm

from longling.framework.ML.MXnet.util import split_and_load

from .parameters import Parameters
from .sym import NetName


class Module(object):
    """
    模块模板
    train 修改流程

    # 1
    修改 __init__ 和 params 方法
    Module(....) 初始化一些通用的参数，比如模型的存储路径等

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

    def __init__(self, parameters):
        """

        Parameters
        ----------
        parameters: Parameters

        """
        # 初始化一些通用的参数
        self.params = parameters
        self.prefix = os.path.join(self.params.model_dir, self.params.model_name)
        self.logger = parameters.logger

        self.sym_gen = NetName

    def __str__(self):
        """
        显示模块参数
        Display the necessary params of this Module

        Returns
        -------

        """
        string = ["Params"]
        for k, v in vars(self.params).items():
            string.append("%s: %s" % (k, v))
        return "\n".join(string)

    @staticmethod
    def load_net(filename, net, ctx=mx.cpu(), allow_missing=False, ignore_extra=False):
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
        allow_missing: bool
        ignore_extra: bool

        Returns
        -------
        The initialized net
        """
        # 根据文件名装载已有的网络参数
        if not os.path.isfile(filename):
            raise FileExistsError
        net.load_parameters(filename, ctx, allow_missing=allow_missing, ignore_extra=ignore_extra)
        return net

    def load(self, net, epoch, ctx=mx.cpu(), allow_missing=False, ignore_extra=False):
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
        allow_missing: bool
        ignore_extra: bool

        Returns
        -------
        The initialized net
        """
        # 根据起始轮次装载已有的网络参数
        filename = self.epoch_params_filename(epoch)
        return self.load_net(filename, net, ctx, allow_missing=allow_missing, ignore_extra=ignore_extra)

    def save(self, suffix=".v{}".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")), model_loop=True):
        old_files = os.listdir(self.params.model_dir)
        for old_file in old_files:
            if ".v" in old_file:
                continue
            if model_loop and old_file == self.epoch_params_filename(self.params.end_epoch):
                shutil.copy(
                    os.path.join(self.params.model_dir, old_file),
                    os.path.join(
                        self.params.model_dir,
                        self.epoch_params_filename(self.params.begin_epoch)
                    )
                )
            os.rename(
                os.path.join(self.params.model_dir, old_file),
                os.path.join(self.params.model_dir, old_file) + suffix,
            )

    def dump_parameters(self, filename=None):
        filename = filename if filename is not None else self.prefix + "-params.yaml"
        self.params.dump(filename, override=True)
        return filename

    def epoch_params_filename(self, epoch):
        return self.prefix + "-%04d.parmas" % epoch

    @staticmethod
    def get_data_iter(params=None):
        # 在这里定义数据加载方法
        return

    # 以下部分定义训练相关的方法
    @staticmethod
    def net_initialize(net, model_ctx, initializer=mx.init.Normal(sigma=.1)):
        """初始化网络参数"""
        net.collect_params().initialize(initializer, ctx=model_ctx)

    @staticmethod
    def get_trainer(net, optimizer='sgd', optimizer_params=None, select=Parameters.train_select):
        """把优化器安装到网络上"""
        trainer = gluon.Trainer(net.collect_params(select), optimizer, optimizer_params)
        return trainer

    @staticmethod
    def save_params(filename, net, select=Parameters.save_select):
        import re
        from mxnet import ndarray
        params = net._collect_params_with_prefix()
        if select:
            pattern = re.compile(select)
            params = {name: value for name, value in params.items() if pattern.match(name)}
        arg_dict = {key: val._reduce() for key, val in params.items()}
        ndarray.save(filename, arg_dict)

    def fit(
            self,
            net, begin_epoch, end_epoch, batch_size,
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
        end_epoch: int
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
        return self.epoch_loop(self.batch_loop(self.fit_f))(
            net=net, begin_epoch=begin_epoch, end_epoch=end_epoch, batch_size=batch_size,
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
                net, begin_epoch, end_epoch, batch_size,
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
            end_epoch: int
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
            for epoch in range(begin_epoch, end_epoch):
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
                # test_eval_res = Module.eval(net, test_data, evaluator=evaluator, ctx=ctx)
                # print(evaluator.format_eval_res(epoch, test_eval_res, loss_values, train_time,
                #                                 logger=evaluator.logger, log_f=evaluator.log_f)[0])

                # todo 定义模型保存方案
                if kwargs.get('prefix') and (epoch % kwargs.get('save_epoch', 1) == 0
                                             or end_epoch - 10 <= epoch <= end_epoch - 1):
                    Module.save_params(kwargs['prefix'] + "-%04d.parmas" % (epoch + 1), net)

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
            for i, batch_data in enumerate(train_data):
                fit_f(
                    net=net, batch_size=batch_size, batch_data=batch_data,
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
    def eval(net, test_data, evaluator=None, ctx=mx.cpu()):
        """
        在这里定义数据评估方法

        Parameters
        ----------
        net
        test_data
        evaluator
        ctx

        Returns
        -------
        metrics: dict

        """
        # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        evalue = []
        evalue_func = lambda p, y: 1 / 2 * (y - p) ** 2

        for batch_data in tqdm(test_data, "evaluating"):
            ctx_data = split_and_load(
                ctx, *batch_data,
                even_split=False
            )
            for (data, label) in ctx_data:
                output = net(data)
                evalue.extend(evalue_func(output, label).asnumpy().tolist())
        return {"evaluation_name": sum(evalue) / len(evalue)}

    @staticmethod
    def fit_f(net, batch_size, batch_data,
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
        batch_data: Iterable
            The batch data for train
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
        ctx_data = Module.split_and_load(
            ctx, *batch_data,
            even_split=False
        )

        bp_loss = None
        with autograd.record():
            # todo modify the component extracted from ctx_data
            for (data, label) in ctx_data:
                # todo modify the input to net
                output = net(data)
                for name, func in loss_function.items():
                    # todo modify the input to func
                    loss = func(output, label)
                    if name in bp_loss_f:
                        bp_loss = loss
                    loss_value = nd.mean(loss).asscalar()
                    if losses_monitor:
                        losses_monitor.update(name, loss_value)

            assert bp_loss is not None
            bp_loss.backward()
        trainer.step(batch_size)

    @staticmethod
    def split_and_load(ctx, *args, **kwargs):
        return split_and_load(ctx, *args, **kwargs)
