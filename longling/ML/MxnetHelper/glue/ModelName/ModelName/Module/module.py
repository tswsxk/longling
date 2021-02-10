# coding: utf-8
# Copyright @tongshiwei
from __future__ import absolute_import

import functools
import logging
import os

import mxnet as mx
from longling.ML.MxnetHelper.glue import module
from longling.ML.toolkit import EpochEvalFMT as Formatter
from tqdm import tqdm

from .configuration import Configuration
from .sym import get_net, fit_f, eval_f, net_init

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
        ) if hasattr(self.cfg, "model_dir") and hasattr(self.cfg, "model_name") else "model"
        self.logger = configuration.logger

    @functools.wraps(fit_f)
    def fit_f(self, *args, **kwargs):
        return fit_f(*args, **kwargs)

    @functools.wraps(get_net)
    def sym_gen(self, *args, **kwargs):
        return get_net(*args, **kwargs)

    @functools.wraps(net_init)
    def net_initialize(self, *args, **kwargs):
        return net_init(*args, **kwargs)

    def dump_configuration(self, filename=None):
        filename = filename if filename is not None \
            else os.path.join(self.cfg.cfg_path)
        self.cfg.dump(filename, override=True)
        return filename

    def epoch_params_filepath(self, epoch):
        return self.prefix + "-%04d.parmas" % epoch

    # 部分定义训练相关的方法
    @staticmethod
    @functools.wraps(module.Module.get_trainer)
    def get_trainer(
            net, optimizer='sgd', optimizer_params=None, lr_params=None,
            select=Configuration.train_select, logger=logging, *args, **kwargs
    ):
        return module.Module.get_trainer(
            net=net,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            lr_params=lr_params,
            select=select,
            logger=logger,
            *args,
            **kwargs,
        )

    def save_params(self, filename, net):
        module.Module.save_params(filename, net, select=self.cfg.save_select)

    def fit(
            self,
            net, begin_epoch, end_epoch, batch_size,
            train_data,
            trainer,
            loss_function,
            eval_data=None,
            ctx=mx.cpu(),
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
        batch_size: int
                The size of each batch
        train_data: Iterable
            The data used for this train procession,
            NOTICE: should have been divided to batches
        trainer:
            The trainer used to update the parameters of the net
        loss_function: dict of function
            The functions to compute the loss for the procession
            of back propagation
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
            batch_size=batch_size,
            train_data=train_data,
            trainer=trainer,
            loss_function=loss_function,
            test_data=eval_data,
            ctx=ctx,
            toolbox=toolbox,
            **kwargs
        )

    def epoch_loop(
            self,
            net, begin_epoch, end_epoch, batch_size,
            train_data,
            trainer,
            loss_function,
            test_data=None,
            ctx=mx.cpu(),
            toolbox=None,
            save_model=True,
            eval_every_n_epoch=1,
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
        batch_size: int
            The size of each batch
        train_data: Iterable
            The data used for this train procession,
            NOTICE: should have been divided to batches
        trainer:
            The trainer used to update the parameters of the net
        loss_function: dict of function
            The functions to compute the loss for the procession
            of back propagation
        test_data: Iterable
            The data used for the evaluation at the end of each epoch,
            NOTICE: should have been divided to batches
            Default to ``None``
        ctx: Context or list of Context
            Defaults to ``mx.cpu()``.
        toolbox: Toolbox
            Default to ``None``
        save_model: bool
            Whether save model
        eval_every_n_epoch: int
        kwargs
        """
        # 参数修改时需要同步修改 fit 函数中的参数
        # 定义轮次训练过程
        if toolbox is not None:
            formatter = toolbox.get('formatter')
        else:
            formatter = None

        for epoch in range(begin_epoch, end_epoch):
            batch_num, loss_values = self.batch_loop(
                net=net, epoch=epoch, batch_size=batch_size,
                train_data=train_data,
                trainer=trainer,
                loss_function=loss_function,
                ctx=ctx,
                toolbox=toolbox,
            )
            if hasattr(self.cfg, "lr_params") and self.cfg.lr_params \
                    and "update_params" in self.cfg.lr_params and self.cfg.end_epoch - self.cfg.begin_epoch - 1 > 0:
                self.cfg.logger.info("reset trainer")
                lr_params = self.cfg.lr_params.pop("update_params")
                lr_update_params = dict(
                    batches_per_epoch=batch_num,
                    lr=self.cfg.optimizer_params["learning_rate"],
                    update_epoch=lr_params.get(
                        "update_epoch",
                        self.cfg.end_epoch - self.cfg.begin_epoch - 1
                    )
                )
                lr_update_params.update(lr_params)

                trainer = module.Module.get_trainer(
                    net, optimizer=self.cfg.optimizer,
                    optimizer_params=self.cfg.optimizer_params,
                    lr_params=lr_update_params,
                    select=self.cfg.train_select,
                    logger=self.cfg.logger
                )

            try:
                train_time = toolbox["monitor"]["progress"].iteration_time
            except (KeyError, TypeError):
                train_time = None

            if (epoch - 1) % eval_every_n_epoch == 0 or epoch == end_epoch - 1:
                # # todo 定义每一轮结束后的模型评估方法
                evaluation_result = self.eval(
                    net, test_data, ctx=ctx
                )

                evaluation_formatter = formatter.get('evaluation', Formatter()) if formatter else Formatter()

                print(
                    evaluation_formatter(
                        iteration=epoch,
                        train_time=train_time,
                        loss_name_value=loss_values,
                        eval_name_value=evaluation_result,
                        extra_info=None,
                        dump=True,
                        keep="msg",
                    )
                )

            # todo 定义模型保存方案
            if save_model:
                if epoch % kwargs.get('save_epoch', 1) == 0:
                    self.save_params(
                        self.epoch_params_filepath(epoch), net
                    )

    def batch_loop(
            self,
            net, epoch, batch_size,
            train_data,
            trainer,
            loss_function,
            ctx=mx.cpu(),
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
        batch_size: int
            The size of each batch
        train_data: Iterable
            The data used for this train procession,
            NOTICE: should have been divided to batches
        trainer:
            The trainer used to update the parameters of the net
        loss_function: dict of function
            The functions to compute the loss for the procession
            of back propagation
        ctx: Context or list of Context
            Defaults to ``mx.cpu()``.
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
            if progress_monitor is None:
                def progress_monitor(x, y):
                    return tqdm(x, "[%s]" % y)

        for i, batch_data in progress_monitor(enumerate(train_data), epoch):
            self.fit_f(
                net=net, batch_size=batch_size, batch_data=batch_data,
                trainer=trainer,
                loss_function=loss_function,
                loss_monitor=loss_monitor,
                ctx=ctx,
            )

        if loss_monitor is not None:
            loss_values = {
                name: loss for name, loss in loss_monitor.items()
            }
            # optional
            loss_monitor.reset()
        else:
            loss_values = {}
        return i, loss_values

    @staticmethod
    def eval(net, test_data, ctx=mx.cpu()):
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
