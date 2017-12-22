# coding: utf-8
# created by tongshiwei on 17-11-6
import json
import os

import numpy as np

from longling.framework.ML.mxnet.nn.shared.nn import NN
from longling.framework.ML.mxnet.nn.shared.mxDataIterator import TextIterator
from longling.framework.ML.mxnet.nn.shared.text_lib import conv_w2id


from longling.framework.ML.mxnet.nn.dnn import text_dnn


class textDNN(NN):
    '''
    example:


    '''

    def __init__(self, sentence_size, model_dir, vecdict_info,
                 num_label=2, num_hiddens=[100], logger=None):
        super(textDNN, self).__init__(logger)

        self.sentence_size = sentence_size

        self.model_dir = model_dir
        self.checkDir(self.model_dir)

        self.num_label = num_label
        self.num_hiddens = num_hiddens
        self.location_vec = vecdict_info['location_vec']
        self.vocab_size = vecdict_info['vocab_size']
        self.vec_size = vecdict_info['vec_size']

        self.logger.info('sentence_size: %s' % self.sentence_size)
        self.logger.info('model_dir: %s' % self.model_dir)
        self.logger.info('num_label: %s' % num_label)
        self.logger.info('num_hiddens: %s' % self.num_hiddens)
        self.logger.info('location_vec: %s' % self.location_vec)
        self.logger.info('vocab_size: %s' % self.vocab_size)
        self.logger.info('vec_size: %s' % self.vec_size)

    def get_symbol_without_loss(self, batch_size, dropout):
        return text_dnn.get_text_dnn_symbol_without_loss(
            vocab_size=self.sentence_size,
            vec_size=self.vec_size,
            num_label=self.num_label,
            num_hiddens=self.num_hiddens,
            dropout=dropout,
        )

    def get_symbol(self, batch_size, dropout):
        return text_dnn.get_text_dnn_symbol(
            vec_size=self.vec_size,
            vocab_size=self.vocab_size,
            num_label=self.num_label,
            num_hiddens=self.num_hiddens,
            dropout=dropout,
        )

    def get_model(self, batch_size, dropout, ctx=-1, embedding=None, checkpoint=None):
        ctx = self.form_ctx(ctx)

        checkpoint = self.form_checkpoint(self.model_dir, checkpoint)

        return text_dnn.get_text_dnn_model(
            ctx=ctx,
            dnn_symbol=self.get_symbol(None, dropout=dropout),
            embedding=embedding,
            sentence_size=self.sentence_size,
            batch_size=batch_size,
            checkpoint=checkpoint,
        )

    def set_predictor(self, batch_size, ctx, checkpoint, pre_embeding=False, vecdict=None):
        # set w2id dict
        assert vecdict is not None
        self.predictor = {'vecdict': vecdict.w2id}

        self.predictor['batch_size'] = batch_size

        # set predictor nn
        self.predictor['predictor'] = self.get_model(
            batch_size=batch_size,
            dropout=0.0,
            ctx=ctx,
            embedding=vecdict.id2v,
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

        s['sentence_size'] = self.sentence_size
        s['vocab_size'] = self.vocab_size
        s['vec_size'] = self.vec_size
        s['location_vec'] = self.location_vec
        s['num_label'] = self.num_label
        s['num_hiddens'] = self.num_hiddens

        line = json.dumps(s)
        with open(location_parameters, mode='w') as wf:
            wf.write(line)

        self.logger.info("parameters information saved to %s" % location_parameters)

    def network_plot(self, batch_size, node_attrs={}, dropout=0.0, show_tag=False):
        textDNN.plot_network(
            nn_symbol=self.get_symbol(batch_size=batch_size, dropout=dropout),
            save_path=os.path.join(self.model_dir, "plot/network"),
            shape={'data': (batch_size, self.vec_size)},
            node_attrs=node_attrs,
            show_tag=show_tag,
        )

    def process_fit(self, location_train, location_test, vecdict, ctx, parameters={}, epoch=20):
        logger = self.logger
        batch_size = parameters.get('batch_size', 128)
        dropout = parameters.get('dorpout', 0.5)
        checkpoint = parameters.get('checkpoint', None)
        start_epoch = 0 if checkpoint is None else int(checkpoint)

        train_iter = TextIterator(vecdict.w2id, location_train, self.sentence_size)
        test_iter = TextIterator(vecdict.w2id, location_test, self.sentence_size)
        logger.info('data: train-%s, test-%s', train_iter.cnt, test_iter.cnt)

        dnn_model = self.get_model(
            batch_size=batch_size,
            dropout=dropout,
            ctx=ctx,
            embedding=vecdict.embedding,
            checkpoint=checkpoint
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