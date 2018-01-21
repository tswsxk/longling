# coding: utf-8
# create by tongshiwei on 2017/10/28

import json
import os

import numpy as np

from longling.framework.ML.mxnet.nn.shared.nn import NN
from longling.framework.ML.mxnet.nn.shared.mxDataIterator import getNumIterator
from longling.framework.ML.mxnet.nn.shared.text_lib import conv_w2id


from longling.framework.ML.mxnet.nn.dnn import numerical_dnn


class numericalDNN(NN):
    def __init__(self, feature_num, model_dir, num_label=2, num_hiddens=[100], logger=None):
        super(numericalDNN, self).__init__(logger)

        self.feature_num = feature_num

        self.model_dir = model_dir
        self.checkDir(self.model_dir)

        self.num_label = num_label
        self.num_hiddens = num_hiddens

        self.logger.info('feature_num: %s' % self.feature_num)
        self.logger.info('model_dir: %s' % self.model_dir)
        self.logger.info('num_label: %s' % num_label)
        self.logger.info('num_hiddens: %s' % self.num_hiddens)

    def get_symbol_without_loss(self, batch_size, dropout, inner_dropouts=None):
        return numerical_dnn.get_numerical_dnn_symbol_without_loss(
            num_label=self.num_label,
            num_hiddens=self.num_hiddens,
            dropout=dropout,
            inner_dropouts=inner_dropouts,
        )

    def get_symbol(self, dropout, inner_dropouts=None):
        return numerical_dnn.get_numerical_dnn_symbol(
            num_label=self.num_label,
            num_hiddens=self.num_hiddens,
            dropout=dropout,
            inner_dropouts=inner_dropouts,
        )

    def get_model(self, batch_size, dropout, inner_dropouts, ctx=-1, checkpoint=None):
        ctx = self.form_ctx(ctx)

        checkpoint = self.form_checkpoint(self.model_dir, checkpoint)

        return numerical_dnn.get_numerical_dnn_model(
            ctx=ctx,
            dnn_symbol=self.get_symbol(dropout=dropout, inner_dropouts=inner_dropouts),
            feature_num=self.feature_num,
            batch_size=batch_size,
            checkpoint=checkpoint,
        )

    def set_predictor(self, batch_size, ctx, checkpoint, pre_embeding=False, vecdict=None):
        # set w2id dict
        assert vecdict is not None
        self.predictor = dict()

        self.predictor['batch_size'] = batch_size

        # set predictor nn
        self.predictor['predictor'] = self.get_model(
            batch_size=batch_size,
            dropout=0.0,
            ctx=ctx,
            checkpoint=checkpoint,
        )

    def predictProba(self, sentences):
        x_vec = conv_w2id(self.predictor['vecdict'], sentences)
        x_vec = np.reshape(x_vec, (x_vec.shape[0], 1, x_vec.shape[1], x_vec.shape[2]))
        self.predictor['predictor'].data[:] = x_vec
        self.predictor['predictor'].model_exec.forward(is_train=False)
        res = self.predictor['predictor'].model_exec.outputs[0].asnumpy()
        return res

    def record_parameters(self, **extend_parameters):
        location_parameters = os.path.join(self.model_dir, 'parameters.txt')
        s = dict()

        s['batch_size'] = extend_parameters['batch_size']
        s['dropout'] = extend_parameters['dropout']
        s['epoch_num'] = extend_parameters['epoch']

        s['feature_num'] = self.feature_num
        s['num_label'] = self.num_label
        s['num_hiddens'] = self.num_hiddens

        line = json.dumps(s)
        with open(location_parameters, mode='w') as wf:
            wf.write(line)

        self.logger.info("parameters information saved to %s" % location_parameters)

    def network_plot(self, batch_size, node_attrs={}, dropout=0.0, inner_dropouts=None, show_tag=False):
        numericalDNN.plot_network(
            nn_symbol=self.get_symbol(dropout=dropout, inner_dropouts=inner_dropouts),
            save_path=os.path.join(self.model_dir, "plot/network"),
            shape={'data': (batch_size, self.feature_num)},
            node_attrs=node_attrs,
            show_tag=show_tag,
        )

    def process_fit(self, location_train, location_test, ctx, parameters={}, epoch=20):
        logger = self.logger
        batch_size = parameters.get('batch_size', 128)
        dropout = parameters.get('dorpout', 0.5)
        inner_dropouts = parameters.get('inner_dropouts', None)
        checkpoint = parameters.get('checkpoint', None)
        start_epoch = 0 if checkpoint is None else int(checkpoint)

        train_iter = getNumIterator(location_train, self.feature_num, logger=self.logger)
        test_iter = getNumIterator(location_test, self.feature_num, logger=self.logger)
        logger.info('data: train-%s, test-%s', train_iter.cnt, test_iter.cnt)

        dnn_model = self.get_model(
            batch_size=batch_size,
            dropout=dropout,
            inner_dropouts=inner_dropouts,
            ctx=ctx,
            checkpoint=checkpoint,
        )

        self.record_parameters(batch_size=batch_size, dropout=dropout, epoch=epoch)

        self.fit(
            model=dnn_model,
            train_iter=train_iter,
            test_iter=test_iter,
            batch_size=batch_size,
            start_epoch=start_epoch,
            epoch=epoch,
            model_dir=self.model_dir,
        )
        train_iter.close()
        test_iter.close()

    def save_model(self, location=""):
        location = os.path.join(self.model_dir, "model_class") if not location else location
        super(numericalDNN, self).save_model(location)

if __name__ == '__main__':
    root = "../../../../../../"
    model_dir = root + "data/numerical_dnn/test/"

    nn = numericalDNN(
        feature_num=28*28,
        model_dir=model_dir,
        num_label=10,
        num_hiddens=[1024, 512, 128, 64, 32],
    )
    # nn.save_model(nn.model_dir + "model.class")
    # nn = NN.load_model(model_dir + "model.class")
    # nn.network_plot(batch_size=128, show_tag=True)
    nn.process_fit(
        location_train=root + "data/image/one_dim/mnist_train",
        location_test=root + "data/image/one_dim/mnist_test",
        ctx=-1,
        epoch=20,
        parameters={'batch_size': 128}
    )

