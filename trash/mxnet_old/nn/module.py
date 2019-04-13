# coding: utf-8
# created by tongshiwei on 18-1-26


import logging
import math
import time

import mxnet as mx
import traceback
import numpy as np
from tqdm import tqdm


# same as mxnet_old.base._as_list
def _as_list(obj):
    """A utility function that converts the argument to a list if it is not already.

    Parameters
    ----------
    obj : object

    Returns
    -------
    If `obj` is a list or tuple, return it. Otherwise, return `[obj]` as a
    single-element list.

    """
    if isinstance(obj, (list, tuple)):
        return obj
    else:
        return [obj]


def fit(mod, train_data, eval_data=None, eval_metric='acc',
        epoch_end_callback=None, batch_end_callback=None, kvstore='local',
        optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
        eval_end_callback=None,
        eval_batch_end_callback=None, initializer=mx.initializer.Uniform(0.01),
        arg_params=None, aux_params=None, allow_missing=False,
        force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
        validation_metric=None, monitor=None,
        # below are my arguments
        time_monitor=None, max_grad_norm=5.0):
    """Trains the module parameters.

            Checkout `Module Tutorial <http://mxnet_old.io/tutorials/basic/module.html>`_ to see
            a end-to-end use-case.

            Parameters
            ----------
            train_data : DataIter
                Train DataIter.
            eval_data : DataIter
                If not ``None``, will be used as validation set and the performance
                after each epoch will be evaluated.
            eval_metric : str or EvalMetric
                Defaults to 'accuracy'. The performance measure used to display during training.
                Other possible predefined metrics are:
                'ce' (CrossEntropy), 'f1', 'mae', 'mse', 'rmse', 'top_k_accuracy'.
            epoch_end_callback : function or list of functions
                Each callback will be called with the current `epoch`, `symbol`, `arg_params`
                and `aux_params`.
            batch_end_callback : function or list of function
                Each callback will be called with a `BatchEndParam`.
            kvstore : str or KVStore
                Defaults to 'local'.
            optimizer : str or Optimizer
                Defaults to 'sgd'.
            optimizer_params : dict
                Defaults to ``(('learning_rate', 0.01),)``. The parameters for
                the optimizer constructor.
                The default value is not a dict, just to avoid pylint warning on dangerous
                default values.
            eval_end_callback : function or list of function
                These will be called at the end of each full evaluation, with the metrics over
                the entire evaluation set.
            eval_batch_end_callback : function or list of function
                These will be called at the end of each mini-batch during evaluation.
            initializer : Initializer
                The initializer is called to initialize the module parameters when they are
                not already initialized.
            arg_params : dict
                Defaults to ``None``, if not ``None``, should be existing parameters from a trained
                model or loaded from a checkpoint (previously saved model). In this case,
                the value here will be used to initialize the module parameters, unless they
                are already initialized by the user via a call to `init_params` or `fit`.
                `arg_params` has a higher priority than `initializer`.
            aux_params : dict
                Defaults to ``None``. Similar to `arg_params`, except for auxiliary states.
            allow_missing : bool
                Defaults to ``False``. Indicates whether to allow missing parameters when `arg_params`
                and `aux_params` are not ``None``. If this is ``True``, then the missing parameters
                will be initialized via the `initializer`.
            force_rebind : bool
                Defaults to ``False``. Whether to force rebinding the executors if already bound.
            force_init : bool
                Defaults to ``False``. Indicates whether to force initialization even if the
                parameters are already initialized.
            begin_epoch : int
                Defaults to 0. Indicates the starting epoch. Usually, if resumed from a
                checkpoint saved at a previous training phase at epoch N, then this value should be
                N+1.
            num_epoch : int
                Number of epochs for training.
            validation_metric ：
                Defaults to None.
            monitor :
                Defaults to None.
            # below are my arguments
            time_monitor : TimeMonitor
               Defaults to  None.
            max_grad_norm : float
                Defaults to 0.0
            Examples
            --------
            >>> # An example of using fit for training.
            >>> # Assume training dataIter and validation dataIter are ready
            >>> # Assume loading a previously checkpointed model
            >>> sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 3)
            >>> mod.fit(train_data=train_dataiter, eval_data=val_dataiter, optimizer='sgd',
            ...     optimizer_params={'learning_rate':0.01, 'momentum': 0.9},
            ...     arg_params=arg_params, aux_params=aux_params,
            ...     eval_metric='acc', num_epoch=10, begin_epoch=3)
            """
    assert num_epoch is not None, 'please specify number of epochs'

    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
             for_training=True, force_rebind=force_rebind)
    if monitor is not None:
        mod.install_monitor(monitor)
    mod.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                    allow_missing=allow_missing, force_init=force_init)
    mod.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                       optimizer_params=optimizer_params)

    if validation_metric is None:
        validation_metric = eval_metric
    if not isinstance(eval_metric, mx.metric.EvalMetric):
        eval_metric = mx.metric.create(eval_metric)

    ################################################################################
    # training loop
    ################################################################################

    for epoch in range(begin_epoch, num_epoch):  # for epoch in range(begin_epoch, num_epoch):
        tic = time.time()
        eval_metric.reset()
        nbatch = 0

        data_iter = iter(train_data)
        end_of_batch = False
        next_data_batch = next(data_iter)

        while not end_of_batch:
            data_batch = next_data_batch
            if monitor is not None:
                monitor.tic()

            mod.forward_backward(data_batch)
            mod.update()

            try:
                # pre fetch next batch
                next_data_batch = next(data_iter)
                mod.prepare(next_data_batch)
            except StopIteration:
                end_of_batch = True

            mod.update_metric(eval_metric, data_batch.label)

            if monitor is not None:
                monitor.toc_print()

            if batch_end_callback is not None:
                batch_end_params = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                          eval_metric=eval_metric,
                                                          locals=locals())
                for callback in _as_list(batch_end_callback):
                    callback(batch_end_params)
            nbatch += 1

        # one epoch of training is finished
        for name, val in eval_metric.get_name_value():
            mod.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
        toc = time.time()
        mod.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))

        # sync aux params across devices
        arg_params, aux_params = mod.get_params()
        mod.set_params(arg_params, aux_params)

        if epoch_end_callback is not None:
            for callback in _as_list(epoch_end_callback):
                callback(epoch, mod.symbol, arg_params, aux_params)

        # ----------------------------------------
        # evaluation on validation set
        if eval_data:
            res = mod.score(eval_data, validation_metric,
                            score_end_callback=eval_end_callback,
                            batch_end_callback=eval_batch_end_callback, epoch=epoch)
            # TODO: pull this into default
            for name, val in res:
                mod.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

        # end of 1 epoch, reset the data-iter for another epoch
        train_data.reset()
