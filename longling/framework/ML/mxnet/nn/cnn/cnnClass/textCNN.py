# coding: utf-8
# create by tongshiwei on 2017/10/22
import json
import os

import mxnet as mx
import numpy as np

from longling.framework.ML.mxnet.nn.shared.nn import NN, add_loss_layer
from longling.framework.ML.mxnet.nn.shared.mxDataIterator import TextIterator
from longling.framework.ML.mxnet.nn.shared.text_lib import conv_w2id, conv_w2v

from longling.framework.ML.mxnet.nn.cnn import text_cnn


class textCNN(NN):
    '''
    example:


    '''

    def __init__(self, sentence_size, model_dir, vecdict_info,
                 num_label=2, filter_list=[1, 2, 3, 4], num_filter=60, logger=None):
        super(textCNN, self).__init__(logger)

        self.sentence_size = sentence_size

        self.model_dir = model_dir
        self.checkDir(self.model_dir)

        self.num_label = num_label
        self.filter_list = filter_list
        self.num_filter = num_filter
        self.location_vec = vecdict_info['location_vec']
        self.vocab_size = vecdict_info['vocab_size']
        self.vec_size = vecdict_info['vec_size']

        self.logger.info('sentence_size: %s' % self.sentence_size)
        self.logger.info('model_dir: %s' % self.model_dir)
        self.logger.info('num_label: %s' % num_label)
        self.logger.info('filter_list: %s' % filter_list)
        self.logger.info('num_filter: %s' % self.num_filter)
        self.logger.info('location_vec: %s' % self.location_vec)
        self.logger.info('vocab_size: %s' % self.vocab_size)
        self.logger.info('vec_size: %s' % self.vec_size)

    def get_symbol_without_loss(self, batch_size, dropout):
        return text_cnn.get_text_cnn_symbol_without_loss(
            sentence_size=self.sentence_size,
            vec_size=self.vec_size,
            batch_size=batch_size,
            vocab_size=self.vocab_size,
            num_label=self.num_label,
            filter_list=self.filter_list,
            num_filter=self.num_filter,
            dropout=dropout,
        )

    def get_symbol(self, batch_size, dropout):
        layer = self.get_symbol_without_loss(batch_size, dropout)
        return add_loss_layer(layer)

    def get_model(self, batch_size, dropout, ctx=-1, embedding=None, checkpoint=None):
        ctx = self.form_ctx(ctx)

        checkpoint = self.form_checkpoint(self.model_dir, checkpoint)

        return text_cnn.get_text_cnn_model(
            ctx=ctx,
            cnn_symbol=self.get_symbol(batch_size, dropout=dropout),
            embedding=embedding,
            sentence_size=self.sentence_size,
            batch_size=batch_size,
            checkpoint=checkpoint,
            logger=self.logger,
        )

    def set_predictor(self, batch_size, ctx, checkpoint, vecdict=None):
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
        s['num_filter'] = self.num_filter

        line = json.dumps(s)
        with open(location_parameters, mode='w') as wf:
            wf.write(line)

        self.logger.info("parameters information saved to %s" % location_parameters)

    def network_plot(self, batch_size, node_attrs={}, dropout=0.0, show_tag=False):
        textCNN.plot_network(
            nn_symbol=self.get_symbol(batch_size=batch_size, dropout=dropout),
            save_path=os.path.join(self.model_dir, "plot/network"),
            shape={'data': (batch_size, self.sentence_size)},
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

        cnn_model = self.get_model(
            batch_size=batch_size,
            dropout=dropout,
            ctx=ctx,
            embedding=vecdict.embedding,
            checkpoint=checkpoint
        )

        self.record_parameters(batch_size=batch_size, dropout=dropout, epoch=epoch)

        self.fit(
            model=cnn_model,
            train_iter=train_iter,
            test_iter=test_iter,
            batch_size=batch_size,
            start_epoch=start_epoch,
            epoch=epoch,
            model_dir=self.model_dir,
        )


class textKeywordCNN(textCNN):
    def get_symbol(self, batch_size, dropout):
        symbol = self.get_symbol_without_loss(
            batch_size=batch_size,
            dropout=dropout,
        )
        return mx.symbol.MakeLoss(mx.symbol.max(symbol, axis=1))

    def set_predictor(self, batch_size, ctx, checkpoint, w2v_dict=None):
        # set w2v dict
        assert w2v_dict is not None
        self.predictor = {'vecdict': w2v_dict}

        self.predictor['batch_size'] = batch_size

        # set predictor nn
        symbol = text_cnn.get_text_cnn_symbol(
            sentence_size=self.sentence_size,
            vec_size=self.vec_size,
            batch_size=batch_size,
            vocab_size=None,
            num_label=self.num_label,
            filter_list=self.filter_list,
            num_filter=self.num_filter,
            dropout=0.0,
        )
        self.predictor['predictor'] = text_cnn.get_text_cnn_model_without_embedding(
            ctx=self.form_ctx(ctx),
            cnn_symbol=symbol,
            vec_size=self.vec_size,
            sentence_size=self.sentence_size,
            batch_size=batch_size,
            return_grad=True,
            checkpoint=self.form_checkpoint(self.model_dir, checkpoint),
            logger=self.logger,
        )

    def predictProba(self, sentences):
        x_vec = conv_w2v(self.predictor['vecdict'], sentences)
        x_vec = np.reshape(x_vec, (x_vec.shape[0], 1, x_vec.shape[1], x_vec.shape[2]))
        self.predictor['predictor'].data[:] = x_vec
        self.predictor['predictor'].model_exec.forward(is_train=False)
        res = self.predictor['predictor'].model_exec.outputs[0].asnumpy()
        return res

    def get_grad(self, sentences, map_range=None, center_tag=False, two_dir_tag=False, tail=0, empty_shift=0,
                 valid_pad=0):
        x_vec = conv_w2v(self.predictor['vecdict'], sentences)
        x_vec = np.reshape(x_vec, (x_vec.shape[0], 1, x_vec.shape[1], x_vec.shape[2]))

        self.predictor['predictor'].data[:] = x_vec
        self.predictor['predictor'].model_exec.forward(is_train=False)
        self.predictor['predictor'].model_exec.backward()
        grad_map = self.predictor['predictor'].model_exec.args_grad['data'].asnumpy() * x_vec

        return text_cnn.get_keyword_maps(
            grad_map=grad_map,
            sentences=sentences,
            map_range=map_range,
            center_tag=center_tag,
            two_dir_tag=two_dir_tag,
            tail=tail,
            empty_shift=empty_shift,
            valid_pad=valid_pad,
        )


if __name__ == '__main__':
    from longling.framework.ML.mxnet.nn.shared.text_lib import vecDict

    root = "../../../../../../../"
    model_dir = root + "data/text_cnn/test/"
    vecdict = vecDict(root + "data/word2vec/comment.vec.dat")
    nn = textCNN(
        sentence_size=25,
        model_dir=model_dir,
        vecdict_info=vecdict.info,
    )
    nn.network_plot(batch_size=128, show_tag=True)
    # nn.save_model(nn.model_dir + "model.class")
    # nn = NN.load_model(model_dir + "model.class")
    # nn.process_fit(
    #     location_train=root + "data/text/mini.instance.train",
    #     location_test=root + "data/text/mini.instance.test",
    #     vecdict=vecdict,
    #     ctx=-1,
    #     epoch=20,
    # )
    # nn.set_predictor(1, -1, 19, vecdict)
    # print nn.predict([u"wa ka ka ka ka"])
    # nn.clean_predictor()
