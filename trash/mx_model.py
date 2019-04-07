# coding:utf-8
# created by tongshiwei on 2018/7/9
import logging
import time

from collections import OrderedDict

import mxnet as mx
from mxnet import context as ctx

from longling.lib.candylib import as_list

_as_list = as_list

from longling.framework.ML.MXnet.metric import PairwiseMetric


class PairWiseModule(mx.mod.Module):
    def __init__(self,
                 symbol, data_names=('data',), label_names=('label',),
                 pairwise_symbol=None, pair_data_names=('data1', 'data2'),
                 logger=logging, context=ctx.cpu(), work_load_list=None,
                 fixed_param_names=None, state_names=None, group2ctxs=None,
                 compression_params=None):
        super(PairWiseModule, self).__init__(
            symbol=symbol,
            data_names=data_names, label_names=label_names,
            logger=logger, context=context, work_load_list=work_load_list,
            fixed_param_names=fixed_param_names, state_names=state_names, group2ctxs=group2ctxs,
            compression_params=compression_params
        )

        self.pair_module = mx.mod.Module(
            symbol=pairwise_symbol,
            data_names=pair_data_names, label_names=[],
            logger=logger, work_load_list=work_load_list, fixed_param_names=fixed_param_names,
            state_names=state_names, group2ctxs=group2ctxs, compression_params=compression_params,
        ) if pairwise_symbol is not None else None
        self.pairwise_symbol = pairwise_symbol

    def pairwise_fit(self, train_data, eval_data=None, eval_metric=PairwiseMetric(),
                     epoch_end_callback=None, batch_end_callback=None, kvstore='local',
                     optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
                     eval_end_callback=None,
                     eval_batch_end_callback=None, initializer=mx.init.Uniform(0.01),
                     arg_params=None, aux_params=None, allow_missing=False,
                     force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
                     validation_metric=None, monitor=None, sparse_row_id_fn=None,
                     *args, **kwargs):
        assert self.pairwise_symbol is not None and self.pair_module is not None, \
            'please specify pairwise symbol'
        assert num_epoch is not None, 'please specify number of epochs'

        self.pair_module.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                              for_training=True, force_rebind=force_rebind)

        if monitor is not None:
            self.pair_module.install_monitor(monitor)
        self.pair_module.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                                     allow_missing=allow_missing, force_init=force_init)
        self.pair_module.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                                        optimizer_params=optimizer_params)

        if eval_data is not None and validation_metric is None:
            raise AssertionError("when eval data is not None, validation_metric should not be None")
        if not isinstance(eval_metric, mx.metric.EvalMetric):
            eval_metric = mx.metric.create(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        for epoch in range(begin_epoch, num_epoch):
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
                self.pair_module.forward_backward(data_batch)
                self.pair_module.update()
                try:
                    # pre fetch next batch
                    next_data_batch = next(data_iter)
                    self.pair_module.prepare(next_data_batch, sparse_row_id_fn=sparse_row_id_fn)
                except StopIteration:
                    end_of_batch = True

                if monitor is not None:
                    monitor.toc_print()

                for texec, islice in zip(self.pair_module._exec_group.execs, self.pair_module._exec_group.slices):
                    preds = OrderedDict(zip(self.output_names, texec.outputs))
                    eval_metric.update_dict({None: []}, preds)

                if end_of_batch:
                    eval_name_vals = eval_metric.get_name_value()

                if batch_end_callback is not None:
                    batch_end_params = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                              eval_metric=eval_metric,
                                                              locals=locals())
                    for callback in _as_list(batch_end_callback):
                        callback(batch_end_params)
                nbatch += 1

            # one epoch of training is finished
            for name, val in eval_name_vals:
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc - tic))

            # sync aux params across devices
            arg_params, aux_params = self.pair_module.get_params()
            self.pair_module.set_params(arg_params, aux_params)

            if epoch_end_callback is not None:
                for callback in _as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)

            # ----------------------------------------
            # evaluation on validation set
            if eval_data:
                self.bind(data_shapes=eval_data.provide_data, label_shapes=eval_data.provide_label,
                          for_training=True, force_rebind=force_rebind)
                arg_params, aux_params = self.pair_module.get_params()
                self.set_params(arg_params, aux_params)

                res = self.score(eval_data, validation_metric,
                                 score_end_callback=eval_end_callback,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                # TODO: pull this into default
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()
