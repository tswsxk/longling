# coding: utf-8
# created by tongshiwei on 17-11-5
import json
import os

from longling.framework.ML.mxnet.nn.shared.nn import NN, add_loss_layer
from longling.framework.ML.mxnet.nn.shared.mxDataIterator import getNumIterator

from longling.framework.ML.mxnet.nn.cnn import numerical_cnn


class numericalCNN(NN):
    '''
    example:


    '''

    def __init__(self, feature_num, vec_size, model_dir, channel=1,
                 num_label=2, kernel_list=[[(5, 5), (5, 5)]], pool_list=[[(2, 2), (2, 2)]], fc_list=[[84]],
                 num_filter_list=[[6, 16]],
                 logger=None):
        super(numericalCNN, self).__init__(logger)

        self.channel = channel
        self.feature_num = feature_num
        self.vec_size = vec_size

        self.model_dir = model_dir
        self.checkDir(self.model_dir)

        self.num_label = num_label
        self.kernel_list = kernel_list
        self.pool_list = pool_list
        self.fc_list = fc_list
        self.num_filter_list = num_filter_list

        self.logger.info('channel: %s' % self.channel)
        self.logger.info('feature_num: %s' % self.feature_num)
        self.logger.info('vec_size: %s' % self.vec_size)
        self.logger.info('model_dir: %s' % self.model_dir)
        self.logger.info('num_label: %s' % num_label)
        self.logger.info('kernal_list: %s' % self.kernel_list)
        self.logger.info('pool_list: %s' % self.pool_list)
        self.logger.info('fc_list: %s' % self.fc_list)
        self.logger.info('num_filter: %s' % self.num_filter_list)

    def get_symbol_without_loss(self, batch_size, dropout):
        return numerical_cnn.get_numerical_cnn_symbol_without_loss(
            batch_size=batch_size,
            channel=self.channel,
            feature_num=self.feature_num,
            vec_size=self.vec_size,
            num_label=self.num_label,
            kernel_list=self.kernel_list,
            pool_list=self.pool_list,
            fc_list=self.fc_list,
            num_filter_list=self.num_filter_list,
            dropout=dropout,
        )

    def get_symbol(self, batch_size, dropout):
        layer = self.get_symbol_without_loss(batch_size, dropout)
        return add_loss_layer(layer)

    def get_model(self, batch_size, dropout, ctx=-1, checkpoint=None):
        ctx = self.form_ctx(ctx)

        checkpoint = self.form_checkpoint(self.model_dir, checkpoint)

        return numerical_cnn.get_numerical_cnn_model(
            ctx=ctx,
            cnn_symbol=self.get_symbol(batch_size, dropout=dropout),
            channel=self.channel,
            feature_num=self.feature_num,
            batch_size=batch_size,
            vec_size=self.vec_size,
            checkpoint=checkpoint,
            logger=self.logger,
        )

    def set_predictor(self, batch_size, ctx, checkpoint):
        # set w2id dict
        self.predictor = dict()

        self.predictor['batch_size'] = batch_size

        # set predictor nn
        self.predictor['predictor'] = self.get_model(
            batch_size=batch_size,
            dropout=0.0,
            ctx=ctx,
            checkpoint=checkpoint,
        )

    def predictProba(self, x_vec):
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
        s['vec_size'] = self.vec_size
        s['num_label'] = self.num_label
        s['num_filter_list'] = self.num_filter_list

        line = json.dumps(s)
        with open(location_parameters, mode='w') as wf:
            wf.write(line)

        self.logger.info("parameters information saved to %s" % location_parameters)

    def network_plot(self, batch_size, node_attrs={}, dropout=0.0, show_tag=False):
        numericalCNN.plot_network(
            nn_symbol=self.get_symbol(batch_size=batch_size, dropout=dropout),
            save_path=os.path.join(self.model_dir, "plot/network"),
            shape={'data': (batch_size, self.channel, self.feature_num, self.vec_size)},
            node_attrs=node_attrs,
            show_tag=show_tag,
        )

    def process_fit(self, location_train, location_test, ctx, parameters={}, epoch=20):
        logger = self.logger
        batch_size = parameters.get('batch_size', 128)
        dropout = parameters.get('dorpout', 0.5)
        checkpoint = parameters.get('checkpoint', None)
        start_epoch = 0 if checkpoint is None else int(checkpoint)

        iter_paras = {'logger': self.logger}
        if parameters.get('reshape', False):
            iter_paras['channel'] = self.channel
            iter_paras['feature_num'] = self.feature_num
            iter_paras['vec_size'] = self.vec_size
        if parameters.get('isString', None) is not None:
            iter_paras['isString'] = parameters['isString']

        train_iter = getNumIterator(location_train, **iter_paras)
        test_iter = getNumIterator(location_test, **iter_paras)

        logger.info('data: train-%s, test-%s', train_iter.cnt, test_iter.cnt)

        cnn_model = self.get_model(
            batch_size=batch_size,
            dropout=dropout,
            ctx=ctx,
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
        train_iter.close()
        test_iter.close()


class lenet5CNN(numericalCNN):
    def __init__(self, model_dir, feature_num=32, vec_size=32, channel=1, num_label=2,
                 kernel_list=[[(5, 5), (5, 5), (5, 5)]],
                 pool_list=[[(2, 2), (2, 2)]], fc_list=[[84]], num_filter_list=[[6, 16, 120]], logger=None):
        super(lenet5CNN, self).__init__(
            channel=channel,
            feature_num=feature_num,
            vec_size=vec_size,
            model_dir=model_dir,
            num_label=num_label,
            kernel_list=kernel_list,
            pool_list=pool_list,
            fc_list=fc_list,
            num_filter_list=num_filter_list,
            logger=logger
        )

    def get_symbol_without_loss(self, batch_size, dropout):
        return numerical_cnn.lenet5like_symbol(
            batch_size=batch_size,
            feature_num=self.feature_num,
            vec_size=self.vec_size,
            num_label=self.num_label,
            dropout=dropout,
            kernel_list=self.kernel_list,
            pool_list=self.pool_list,
            fc_list=self.fc_list,
            num_filter_list=self.num_filter_list,

        )


if __name__ == '__main__':
    root = "../../../../../../../"
    model_dir = root + "data/numerical_cnn/test/"
    # nn = lenet5CNN(
    #     model_dir=model_dir,
    # )
    nn = numericalCNN(
        channel=1,
        feature_num=28,
        vec_size=28,
        num_label=10,
        model_dir=model_dir,
    )
    # nn.network_plot(batch_size=128, show_tag=True)
    # nn.save_model(nn.model_dir + "model.class")
    # nn = NN.load_model(model_dir + "model.class")
    nn.process_fit(
        location_train=root + "data/image/mnist_train",
        location_test=root + "data/image/mnist_test",
        ctx=-1,
        epoch=20,
        parameters={'batch_size': 128, 'isString': False, 'reshape': False},
    )
    # nn.set_predictor(1, -1, 19)
    # print nn.predict([u"wa ka ka ka ka"])
    # nn.clean_predictor()
