# coding: utf-8
# Copyright @tongshiwei
from __future__ import absolute_import

import os

from longling.ML.PytorchHelper.tore import module

from .configuration import Configuration
from .sym import get_net, fit_f, eval_f

__all__ = ["Module"]


class Module(module.Module):
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

    def __init__(self, configuration):
        """

        Parameters
        ----------
        configuration: Configuration

        """
        # 初始化一些通用的参数
        self.cfg = configuration
        self.prefix = os.path.join(
            self.cfg.model_dir, self.cfg.model_name
        )
        self.logger = configuration.logger

        self.sym_gen = get_net
        self.fit_f = fit_f

    def dump_configuration(self, filename=None):
        filename = filename if filename is not None \
            else os.path.join(self.cfg.cfg_path)
        self.cfg.dump(filename, override=True)
        return filename

    def epoch_params_filename(self, epoch):
        return self.prefix + "-%04d.parmas" % epoch

    # 部分定义训练相关的方法
    @staticmethod
    def get_trainer(
            net, optimizer='sgd', optimizer_params=None, lr_params=None,
            select=Configuration.train_select
    ):
        return module.Module.get_trainer(
            net, optimizer, optimizer_params, select
        )

    def save_params(self, filename, net):
        module.Module.save_params(filename, net)

    def fit(
            self,
            net, begin_epoch, end_epoch,
            train_data,
            trainer, bp_loss_f,
            loss_function,
            eval_data=None,
            ctx=None,
            toolbox=None,
            **kwargs
    ):
        """
        API for train

        Parameters
        ----------
        net: HybridBlock
            The network which has been initialized or loaded from
            the existed model
        begin_epoch: int
            The begin epoch of this train procession
        end_epoch: int
            The end epoch of this train procession
        train_data: Iterable
            The data used for this train procession,
            NOTICE: should have been divided to batches
        trainer:
            The trainer used to update the parameters of the net
        bp_loss_f: dict with only one value and one key
            The function to compute the loss for the procession
            of back propagation
        loss_function: dict of function
            Some other measurement in addition to bp_loss_f
        eval_data: Iterable
            The data used for the evaluation at the end of each epoch,
            NOTICE: should have been divided to batches
            Default to ``None``
        ctx: Context or list of Context
            Defaults to ``mx.cpu()``.
        toolbox: dict or None
            Default to ``None``
        kwargs

        Returns
        -------

        """
        # 此方法可以直接使用
        return self.epoch_loop(
            net=net, begin_epoch=begin_epoch, end_epoch=end_epoch,
            train_data=train_data,
            trainer=trainer, bp_loss_f=bp_loss_f,
            loss_function=loss_function,
            test_data=eval_data,
            toolbox=toolbox,
            **kwargs
        )

    def epoch_loop(
            self,
            net, begin_epoch, end_epoch,
            train_data,
            trainer,
            bp_loss_f, loss_function,
            test_data=None,
            toolbox=None,
            **kwargs
    ):
        """
        此函数包裹批次训练过程，形成轮次训练过程

        Parameters
        ----------
        net: HybridBlock
            The network which has been initialized or loaded from
            the existed model
        begin_epoch: int
            The begin epoch of this train procession
        end_epoch: int
            The end epoch of this train procession
        train_data: Iterable
            The data used for this train procession,
            NOTICE: should have been divided to batches
        trainer:
            The trainer used to update the parameters of the net
        bp_loss_f: dict with only one value and one key
            The function to compute the loss for the procession
            of back propagation
        loss_function: dict of function
            Some other measurement in addition to bp_loss_f
        test_data: Iterable
            The data used for the evaluation at the end of each epoch,
            NOTICE: should have been divided to batches
            Default to ``None``
        toolbox: Toolbox
            Default to ``None``
        kwargs
        """
        # 参数修改时需要同步修改 fit 函数中的参数
        # 定义轮次训练过程
        if toolbox is not None:
            epoch_timer = toolbox.get('timer')
            formatter = toolbox.get('formatter')
        else:
            epoch_timer = None
            formatter = None

        for epoch in range(begin_epoch, end_epoch):
            if epoch_timer:
                epoch_timer.start()

            loss_values = self.batch_loop(
                net=net, epoch=epoch,
                train_data=train_data,
                trainer=trainer, bp_loss_f=bp_loss_f,
                loss_function=loss_function,
                toolbox=toolbox,
            )

            train_time = epoch_timer.end(wall=True) if epoch_timer else None

            evaluation_result = Module.eval(
                net, test_data
            )
            if formatter:
                evaluation_formatter = formatter.get('evaluation')
                if evaluation_formatter:
                    print(
                        evaluation_formatter(
                            epoch=epoch,
                            train_time=train_time,
                            loss_name_value=loss_values,
                            eval_name_value=evaluation_result,
                            extra_info=None,
                            dump=True,
                        )[0]
                    )

            if kwargs.get('prefix') and (
                    epoch % kwargs.get('save_epoch', 1) == 0 or end_epoch - 10 <= epoch <= end_epoch - 1
            ):
                self.save_params(
                    kwargs['prefix'] + "-%04d.parmas" % (epoch + 1), net
                )

    def batch_loop(
            self,
            net, epoch,
            train_data,
            trainer, bp_loss_f,
            loss_function,
            toolbox=None,
    ):
        """
        The true body of batch loop

        Parameters
        ----------
        net: HybridBlock
            The network which has been initialized
            or loaded from the existed model
        epoch: int
            Current Epoch
        train_data: Iterable
            The data used for this train procession,
            NOTICE: should have been divided to batches
        trainer:
            The trainer used to update the parameters of the net
        bp_loss_f: dict with only one value and one key
            The function to compute the loss for the procession
            of back propagation
        loss_function: dict of function
            Some other measurement in addition to bp_loss_f
        toolbox: dict
            Default to ``None``

        Returns
        -------
        loss_values: dict
            Recorded the loss values computed by the loss_function
        """
        loss_monitor = None
        progress_monitor = None

        if toolbox is not None:
            monitor = toolbox.get("monitor")
            if monitor is not None:
                loss_monitor = monitor.get("loss")
                progress_monitor = monitor.get("progress")

        for i, batch_data in progress_monitor(enumerate(train_data), epoch):
            fit_f(
                net=net, batch_data=batch_data,
                trainer=trainer, bp_loss_f=bp_loss_f,
                loss_function=loss_function,
                loss_monitor=loss_monitor,
            )

        if loss_monitor is not None:
            loss_values = {
                name: loss for name, loss in loss_monitor.items()
            }
            # optional, whether reset the loss at the end of each epoch
            loss_monitor.reset()
        else:
            loss_values = {}

        return loss_values

    @staticmethod
    def eval(net, test_data, ctx=None):
        """
        在这里定义数据评估方法

        Parameters
        ----------
        net
        test_data
        ctx

        Returns
        -------
        metrics: dict

        """
        return eval_f(net, test_data, ctx)
