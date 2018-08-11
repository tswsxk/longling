# coding:utf-8
# created by tongshiwei on 2018/7/13
"""
对gluon的训练、测试过程进行封装
fit_f = epoch_loop(batch_loop(fit_f))
大量使用 staticmethod 并用 get_params 对参数进行分离的原因是因为耦合性太高会导致改起来不太方便
可能修改的地方用 todo 标出
"""
import os

from collections import OrderedDict

import mxnet as mx
from mxnet import autograd, nd
from mxnet import gluon
from mxnet.gluon import utils as gutil

from longling.lib.utilog import config_logging, LogLevel
from longling.lib.clock import Clock
from longling.framework.ML.MXnet.mx_gluon.gluon_toolkit import TrainBatchInformer, MovingLosses
from longling.framework.ML.MXnet.mx_gluon.gluon_toolkit import Evaluator, ClassEvaluator
from longling.framework.ML.MXnet.viz import plot_network, VizError
from longling.framework.ML.MXnet.mx_gluon.gluon_sym import PairwiseLoss, SoftmaxCrossEntropyLoss
from longling.framework.ML.MXnet.mx_gluon.gluon_sym import PVWeight, TextCNN

test_tag = False
view_tag = False
cnn_tag = False
cnn_len = 27
NUM = 0
CTX = [mx.gpu(NUM)]
BATCH_SIZE = 32
MODEL_NAME = 'RLSTM%s' % NUM
LEARNING_RATE = 0.03


########################################################################
# write the user function here
class RLSTM(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(RLSTM, self).__init__(**kwargs)
        self.word_length = None
        self.charater_length = None

    def get_weight(self, name, shape):
        parmas = self.params.get(name, shape=shape)
        var = mx.sym.Variable(name, shape=shape)
        return PVWeight(parmas, var)

    def set_network_unroll(self, word_length, character_length):
        self.word_length = word_length
        self.charater_length = character_length


class RLSTM_CNN3D(RLSTM):
    def __init__(self, lstm_hidden=256, fc_output=32, embedding_dim=256,
                 word_embedding_size=352183, word_radical_embedding_size=31759,
                 char_embedding_size=4746, char_radical_embedding_size=242,
                 **kwargs):
        super(RLSTM_CNN3D, self).__init__(**kwargs)
        self.lstm_hidden = lstm_hidden
        with self.name_scope():
            self.word_embedding = gluon.nn.Embedding(word_embedding_size, embedding_dim)
            self.word_radical_embedding = gluon.nn.Embedding(word_radical_embedding_size, embedding_dim)
            self.char_embedding = gluon.nn.Embedding(char_embedding_size, embedding_dim)
            self.char_radical_embedding = gluon.nn.Embedding(char_radical_embedding_size, embedding_dim)
            # for i in range(4):
            #     setattr(self, "lstms%s" % i, gluon.rnn.LSTMCell(lstm_hidden))
            # self.layers_attention = gluon.nn.Dense(lstm_hidden, activation='tanh')
            # self.batch_norm = gluon.nn.BatchNorm()
            # self.fc = gluon.nn.Dense(fc_output)
            self.text_cnn = TextCNN(cnn_len, embedding_dim, 4, num_output=fc_output, num_highway=False, dropout=0.2)
            # self.dropout = gluon.nn.Dropout(0.5)

    def hybrid_forward(self, F, word_seq, word_radical_seq, character_seq,
                       character_radical_seq, word_mask, charater_mask,
                       *args, **kwargs):
        word_embedding = self.word_embedding(word_seq)
        word_radical_embedding = self.word_radical_embedding(word_radical_seq)
        character_embedding = self.char_embedding(character_seq)
        character_radical_embedding = self.char_radical_embedding(character_radical_seq)

        # word_length = self.word_length
        # character_length = self.charater_length

        # merge_outputs = True
        # w_e, (w_o, w_s) = getattr(self, "lstms0").unroll(word_length, word_embedding, merge_outputs=merge_outputs,
        #                                                  valid_length=word_mask)
        # wr_e, (wr_o, wr_s) = getattr(self, "lstms1").unroll(word_length, word_radical_embedding, merge_outputs=merge_outputs,
        #                                           valid_length=word_mask)
        # c_e, (c_o, c_s) = getattr(self, "lstms2").unroll(character_length, character_embedding, merge_outputs=merge_outputs,
        #                                        valid_length=charater_mask)
        # cr_e, (cr_o, cr_s) = getattr(self, "lstms3").unroll(character_length, character_radical_embedding,
        #                                           merge_outputs=merge_outputs,
        #                                           valid_length=charater_mask)

        w_e = F.SequenceMask(word_embedding, sequence_length=word_mask, use_sequence_length=True, axis=1)
        wr_e = F.SequenceMask(word_radical_embedding, sequence_length=word_mask, use_sequence_length=True, axis=1)
        c_e = F.SequenceMask(character_embedding, sequence_length=charater_mask, use_sequence_length=True, axis=1)
        cr_e = F.SequenceMask(character_radical_embedding, sequence_length=charater_mask, use_sequence_length=True,
                              axis=1)

        attention = F.stack(*[w_e, wr_e, c_e, cr_e], axis=-1)
        attention = self.text_cnn(attention)
        # fc_in = attention
        # self.batch_norm(F.Dropout(self.layers_attention(attention), 0.5))
        return attention


class RLSTM_CNNATN_CROSS(RLSTM):
    def __init__(self, lstm_hidden=256, fc_output=32, embedding_dim=256,
                 word_embedding_size=352183, word_radical_embedding_size=31759,
                 char_embedding_size=4746, char_radical_embedding_size=242,
                 **kwargs):
        super(RLSTM_CNNATN_CROSS, self).__init__(**kwargs)
        self.lstm_hidden = lstm_hidden
        with self.name_scope():
            self.word_embedding = gluon.nn.Embedding(word_embedding_size, embedding_dim)
            self.word_radical_embedding = gluon.nn.Embedding(word_radical_embedding_size, embedding_dim)
            self.char_embedding = gluon.nn.Embedding(char_embedding_size, embedding_dim)
            self.char_radical_embedding = gluon.nn.Embedding(char_radical_embedding_size, embedding_dim)
            for i in range(4):
                setattr(self, "lstms%s" % i, gluon.rnn.ZoneoutCell(gluon.rnn.LSTMCell(lstm_hidden)))
            self.layers_attention = gluon.nn.Dense(lstm_hidden, activation='relu')
            self.fc = gluon.nn.Dense(fc_output)
            # self.text_cnn = TextCNN(lstm_hidden, 4, highway=False, dropout=0.5, num_output=64)
            self.nn = gluon.nn.Dense(64)
            self.dropout = gluon.nn.Dropout(0.5)

    def hybrid_forward(self, F, word_seq, word_radical_seq, character_seq,
                       character_radical_seq, word_mask, charater_mask,
                       *args, **kwargs):
        word_embedding = self.word_embedding(word_seq)
        word_radical_embedding = self.word_radical_embedding(word_radical_seq)
        character_embedding = self.char_embedding(character_seq)
        character_radical_embedding = self.char_radical_embedding(character_radical_seq)

        # word_embedding = self.dropout(word_embedding)
        # word_radical_embedding = self.dropout(word_radical_embedding)
        # character_embedding = self.dropout(character_embedding)
        # character_radical_embedding = self.dropout(character_radical_embedding)

        word_length = self.word_length
        character_length = self.charater_length

        merge_outputs = True
        w_e, (w_o, w_s) = getattr(self, "lstms0").unroll(word_length, word_embedding, merge_outputs=merge_outputs,
                                                         valid_length=word_mask)
        wr_e, (wr_o, wr_s) = getattr(self, "lstms1").unroll(word_length, word_radical_embedding,
                                                            merge_outputs=merge_outputs,
                                                            valid_length=word_mask)
        c_e, (c_o, c_s) = getattr(self, "lstms2").unroll(character_length, character_embedding,
                                                         merge_outputs=merge_outputs,
                                                         valid_length=charater_mask)
        cr_e, (cr_o, cr_s) = getattr(self, "lstms3").unroll(character_length, character_radical_embedding,
                                                            merge_outputs=merge_outputs,
                                                            valid_length=charater_mask)

        # use final state as attention vector
        # ess = [w_e, c_e]
        # ss = [wr_o, cr_o]
        # final_hiddens = []
        # for es, la in zip(ess, ss):
        #     la = F.expand_dims(la, axis=1)
        #     att = F.softmax(F.batch_dot(es, F.swapaxes(la, 1, 2)))
        #     att = F.swapaxes(att, 1, 2)
        #     res = F.batch_dot(att, es)
        #     final_hiddens.append(F.reshape(res, (0, -1)))

        final_hiddens = [w_o, wr_o, c_o, cr_o]
        # attention = F.stack(*final_hiddens, axis=1)
        attention = F.concat(w_o, wr_o, c_o, cr_o)
        # attention = self.text_cnn(F.swapaxes(attention, 1, 2))
        attention = self.dropout(F.Activation(self.nn(attention), act_type='softrelu'))
        fc_in = self.layers_attention(attention)
        return self.fc(fc_in)


class RLSTM_CNNATN(RLSTM):
    def __init__(self, lstm_hidden=256, fc_output=32, embedding_dim=256,
                 word_embedding_size=352183, word_radical_embedding_size=31759,
                 char_embedding_size=4746, char_radical_embedding_size=242,
                 **kwargs):
        super(RLSTM_CNNATN, self).__init__(**kwargs)
        self.lstm_hidden = lstm_hidden
        with self.name_scope():
            self.word_embedding = gluon.nn.Embedding(word_embedding_size, embedding_dim)
            self.word_radical_embedding = gluon.nn.Embedding(word_radical_embedding_size, embedding_dim)
            self.char_embedding = gluon.nn.Embedding(char_embedding_size, embedding_dim)
            self.char_radical_embedding = gluon.nn.Embedding(char_radical_embedding_size, embedding_dim)
            self.lstms = [gluon.rnn.LSTMCell(lstm_hidden) for _ in range(4)]
            for lstm in self.lstms:
                self.register_child(lstm)
            self.layers_attention = gluon.nn.Dense(lstm_hidden, activation='tanh')
            self.batch_norm = gluon.nn.BatchNorm()
            self.fc = gluon.nn.Dense(fc_output)
            self.text_cnn = TextCNN(lstm_hidden, 4)
            self.dropout = gluon.nn.Dropout(0.5)

    def hybrid_forward(self, F, word_seq, word_radical_seq, character_seq,
                       character_radical_seq, word_mask, charater_mask,
                       *args, **kwargs):
        word_embedding = self.word_embedding(word_seq)
        word_radical_embedding = self.word_radical_embedding(word_radical_seq)
        character_embedding = self.char_embedding(character_seq)
        character_radical_embedding = self.char_radical_embedding(character_radical_seq)

        word_length = self.word_length
        character_length = self.charater_length

        merge_outputs = True
        w_e, (w_o, w_s) = self.lstms[0].unroll(word_length, word_embedding, merge_outputs=merge_outputs,
                                               valid_length=word_mask)
        wr_e, (wr_o, wr_s) = self.lstms[1].unroll(word_length, word_radical_embedding, merge_outputs=merge_outputs,
                                                  valid_length=word_mask)
        c_e, (c_o, c_s) = self.lstms[2].unroll(character_length, character_embedding, merge_outputs=merge_outputs,
                                               valid_length=charater_mask)
        cr_e, (cr_o, cr_s) = self.lstms[3].unroll(character_length, character_radical_embedding,
                                                  merge_outputs=merge_outputs,
                                                  valid_length=charater_mask)

        # use final state as attention vector
        ess = [w_e, wr_e, c_e, cr_e]
        ss = [w_s, wr_s, c_s, cr_s]
        final_hiddens = []
        for es, la in zip(ess, ss):
            la = F.expand_dims(la, axis=1)
            att = F.softmax(F.batch_dot(es, F.swapaxes(la, 1, 2)))
            att = F.swapaxes(att, 1, 2)
            res = F.batch_dot(att, es)
            final_hiddens.append(F.reshape(res, (0, -1)))

        attention = F.stack(*final_hiddens, axis=1)
        attention = self.text_cnn(F.swapaxes(attention, 1, 2))
        fc_in = self.batch_norm(self.dropout(self.layers_attention(attention)))
        return self.fc(fc_in)


class RLSTM_HATN(RLSTM):
    def __init__(self, lstm_hidden=256, fc_output=32, embedding_dim=256,
                 word_embedding_size=352183, word_radical_embedding_size=31759,
                 char_embedding_size=4746, char_radical_embedding_size=242,
                 **kwargs):
        super(RLSTM_HATN, self).__init__(**kwargs)
        self.lstm_hidden = lstm_hidden
        with self.name_scope():
            self.word_embedding = gluon.nn.Embedding(word_embedding_size, embedding_dim)
            self.word_radical_embedding = gluon.nn.Embedding(word_radical_embedding_size, embedding_dim)
            self.char_embedding = gluon.nn.Embedding(char_embedding_size, embedding_dim)
            self.char_radical_embedding = gluon.nn.Embedding(char_radical_embedding_size, embedding_dim)
            self.lstms = [gluon.rnn.LSTMCell(lstm_hidden) for _ in range(4)]
            for lstm in self.lstms:
                self.register_child(lstm)
            self.fc = gluon.nn.Dense(fc_output)
            self.loss = gluon.loss.SoftmaxCrossEntropyLoss()
            self.layer0_attention = self.params.get("layer0_attention_weight", shape=(lstm_hidden,))
            self.layer1_attention = self.params.get("layer1_attention_weight", shape=(lstm_hidden,))
            self.layer2_attention = self.params.get("layer2_attention_weight", shape=(lstm_hidden,))
            self.layer3_attention = self.params.get("layer3_attention_weight", shape=(lstm_hidden,))
            self.layers_attention = self.params.get('layers_attention_weight', shape=(4,))

        self.word_length = None
        self.charater_length = None

    def hybrid_forward(self, F, word_seq, word_radical_seq, character_seq,
                       character_radical_seq, word_mask, charater_mask,
                       *args, **kwargs):
        word_embedding = self.word_embedding(word_seq)
        word_radical_embedding = self.word_radical_embedding(word_radical_seq)
        character_embedding = self.char_embedding(character_seq)
        character_radical_embedding = self.char_radical_embedding(character_radical_seq)

        word_length = self.word_length
        character_length = self.charater_length
        merge_outputs = True

        w_e, (w_o, w_s) = self.lstms[0].unroll(word_length, word_embedding, merge_outputs=merge_outputs,
                                               valid_length=word_mask)
        wr_e, (wr_o, wr_s) = self.lstms[1].unroll(word_length, word_radical_embedding, merge_outputs=merge_outputs,
                                                  valid_length=word_mask)
        c_e, (c_o, c_s) = self.lstms[2].unroll(character_length, character_embedding, merge_outputs=merge_outputs,
                                               valid_length=charater_mask)
        cr_e, (cr_o, cr_s) = self.lstms[3].unroll(character_length, character_radical_embedding,
                                                  merge_outputs=merge_outputs,
                                                  valid_length=charater_mask)

        # use hyper parameter as attention vector
        layer_attention = [kwargs['layer%s_attention' % i] for i in range(4)]
        ess = [w_e, wr_e, c_e, cr_e]
        ss = [la for la in layer_attention]
        final_hiddens = []
        for es, la in zip(ess, ss):
            att = F.softmax(F.dot(F.swapaxes(es, 0, 1), la))
            att = F.transpose(att)
            att = F.expand_dims(att, axis=1)
            res = F.batch_dot(att, es)
            final_hiddens.append(F.reshape(res, shape=(0, -1)))

        attention = F.stack(*final_hiddens)
        layers_attention = kwargs['layers_attention']
        fc_in = F.dot(layers_attention, attention)
        return self.fc(fc_in)


#########################################################################
# todo 重命名train_RLSTM函数到需要的模块名
def train_RLSTM():
    # 1 配置参数初始化
    root = "../../../../../"
    model_name = MODEL_NAME
    model_dir = root + "data/gluon/%s/" % model_name

    mod = RLSTMModule(
        model_dir=model_dir,
        model_name=model_name,
        ctx=CTX
    )
    logger = config_logging(logger=model_name, console_log_level=LogLevel.INFO)
    logger.info(str(mod))

    ############################################################################
    # experiment params

    ############################################################################

    # 2 todo 定义网络结构并保存
    vec_root = root + "data/vec/"
    vec_suffix = ".vec.dat"
    from gluonnlp.embedding import TokenEmbedding
    if not test_tag:
        with Clock(logger=logger, tips='loading embedding'):
            from multiprocessing.pool import ThreadPool
            pool = ThreadPool(4)

            p1 = pool.apply_async(TokenEmbedding.from_file, args=(vec_root + "word" + vec_suffix,))
            p2 = pool.apply_async(TokenEmbedding.from_file, args=(vec_root + "word_radical" + vec_suffix,))
            p3 = pool.apply_async(TokenEmbedding.from_file, args=(vec_root + "word_radical" + vec_suffix,))
            p4 = pool.apply_async(TokenEmbedding.from_file, args=(vec_root + "word_radical" + vec_suffix,))

            pool.close()
            pool.join()

            word_embedding = p1.get()
            word_radical_embedding = p2.get()
            char_embedding = p3.get()
            char_radical_embedding = p4.get()

    # 2.1 重新生成
    logger.info("generating symbol")
    if not test_tag:
        net = RLSTMModule.sym_gen(
            word_embedding_size=len(word_embedding.token_to_idx),
            word_radical_embedding_size=len(word_radical_embedding.token_to_idx),
            char_embedding_size=len(char_embedding.token_to_idx),
            char_radical_embedding_size=len(char_radical_embedding.token_to_idx)
        )
    else:
        net = RLSTMModule.sym_gen(
            # word_embedding_size=len(word_embedding.token_to_idx),
            # word_radical_embedding_size=len(word_radical_embedding.token_to_idx),
            # char_embedding_size=len(char_embedding.token_to_idx),
            # char_radical_embedding_size=len(char_radical_embedding.token_to_idx)
        )

    # 2.2 装载已有模型
    # net = mod.load(epoch)
    # net = RLSTMModule.load_net(filename)

    # 5 todo 定义训练相关参数
    begin_epoch = 0
    epoch_num = 200
    batch_size = BATCH_SIZE
    ctx = mod.ctx

    # 3 todo 自行设定网络输入，可视化检查网络
    try:
        logger.info("visualization")
        word_length, character_length = (1, 2) if not cnn_tag else (cnn_len, cnn_len)
        from copy import deepcopy
        viz_net = deepcopy(net)
        viz_net.set_network_unroll(word_length, character_length)
        viz_shape = {
            'word_seq': (batch_size,) + (word_length,),
            'word_radical_seq': (batch_size,) + (word_length,),
            'character_seq': (batch_size,) + (character_length,),
            'character_radical_seq': (batch_size,) + (character_length,),
            # 'label': (batch_size,) + (1, )
        }
        word_seq = mx.sym.var("word_seq")
        word_radical_seq = mx.sym.var("word_radical_seq")
        character_seq = mx.sym.var("character_seq")
        character_radical_seq = mx.sym.var("character_radical_seq")
        word_mask = mx.sym.var("word_mask")
        char_mask = mx.sym.var("char_mask")
        sym = viz_net(word_seq, word_radical_seq, character_seq, character_radical_seq, word_mask, char_mask)
        plot_network(
            nn_symbol=sym,
            save_path=model_dir + "plot/network",
            shape=viz_shape,
            node_attrs={"fixedsize": "false"},
            view=view_tag
        )
    except VizError as e:
        logger.error("error happen in visualization, aborted")
        logger.error(e)

    # 5 todo 定义损失函数
    # bp_loss_f 定义了用来进行 back propagation 的损失函数
    bp_loss_f = {"cross-entropy": gluon.loss.SoftmaxCrossEntropyLoss()}
    loss_function = {

    }
    loss_function.update(bp_loss_f)
    losses_monitor = MovingLosses(loss_function)

    # 5 todo 初始化一些训练过程中的交互信息
    timer = Clock()
    informer = TrainBatchInformer(loss_index=[name for name in loss_function], end_epoch=epoch_num - 1)
    validation_logger = config_logging(
        filename=model_dir + "result.log",
        logger="%s-validation" % model_name,
        mode="w",
        log_format="%(message)s",
    )
    from longling.framework.ML.MXnet.metric import PRF, Accuracy
    evaluator = ClassEvaluator(
        metrics=[PRF(argmax=False), Accuracy(argmax=False)],
        model_ctx=ctx,
        logger=validation_logger,
        log_f=mod.validation_result_file
    )

    # 4 todo 定义数据加载
    data_root = root + "data/"
    train_data_file = data_root + "train"
    test_data_file = data_root + "test"

    logger.info("loading data")
    if not test_tag:
        unknown_token = word_embedding.unknown_token
        train_data = RLSTMModule.get_data_iter(train_data_file, batch_size,
                                               padding=word_embedding.token_to_idx[unknown_token]
                                               )
        unknown_token = char_embedding.unknown_token
        test_data = RLSTMModule.get_data_iter(test_data_file, batch_size,
                                              padding=char_embedding.token_to_idx[unknown_token]
                                              )
    else:
        # unknown_token = word_embedding.unknown_token
        train_data = RLSTMModule.get_data_iter(train_data_file, batch_size,
                                               # padding=word_embedding.token_to_idx[unknown_token]
                                               )
        # unknown_token = char_embedding.unknown_token
        test_data = RLSTMModule.get_data_iter(test_data_file, batch_size,
                                              # padding=char_embedding.token_to_idx[unknown_token]
                                              )

    # 6 todo 训练
    # 直接装载已有模型，确认这一步可以执行的话可以忽略 2 3 4
    logger.info("start training")
    try:
        net = mod.load(net, begin_epoch, ctx)
        logger.info("load params from existing model file %s" % mod.prefix + "-%04d.parmas" % begin_epoch)
    except FileExistsError:
        logger.info("model doesn't exist, initializing")
        import numpy as np
        RLSTMModule.net_initialize(net, ctx)
        if not test_tag:
            net.word_embedding.weight.set_data(word_embedding.idx_to_vec)
            net.word_radical_embedding.weight.set_data(word_radical_embedding.idx_to_vec)
            net.char_embedding.weight.set_data(char_embedding.idx_to_vec)
            net.char_radical_embedding.weight.set_data(char_radical_embedding.idx_to_vec)
    # RLSTMModule.parameters_stabilize(net)
    trainer = RLSTMModule.get_trainer(net)
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
    # RLSTMModule.eval()

    # 7 todo 关闭输入输出流
    # evaluator.close()


class RLSTMModule(object):
    """
    模块模板
    train 修改流程

    # 1
    修改 __init__ 和 params 方法
    RLSTMModule(....) 初始化一些通用的参数，比如模型的存储路径等

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
        net.load_params(filename, ctx)
        return net

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
    def get_data_iter(filename, batch_size, padding=0, num_buckets=100):
        # 在这里定义数据加载方法
        import json
        from tqdm import tqdm
        import numpy as np
        from gluonnlp.data import FixedBucketSampler, PadSequence
        if not test_tag:
            word_feature = []
            word_radical_feature = []
            char_feature = []
            char_radical_feature = []
            features = [word_feature, word_radical_feature, char_feature, char_radical_feature]
            labels = []
            with open(filename) as f:
                for line in tqdm(f, "loading data from %s" % filename):
                    ds = json.loads(line)
                    data, label = ds['x'], ds['z']
                    word_feature.append(data[0])
                    word_radical_feature.append(data[0])
                    char_feature.append(data[0])
                    char_radical_feature.append(data[0])
                    labels.append(label)
        else:
            import random
            length = 20
            word_length = sorted([random.randint(1, length) for _ in range(1000)])
            char_length = sorted([i + random.randint(0, 5) for i in word_length])
            word_feature = [[random.randint(0, length) for _ in range(i)] for i in word_length]
            word_radical_feature = [[random.randint(0, length) for _ in range(i)] for i in word_length]
            char_feature = [[random.randint(0, length) for _ in range(i)] for i in char_length]
            char_radical_feature = [[random.randint(0, length) for _ in range(i)] for i in char_length]

            features = [word_feature, word_radical_feature, char_feature, char_radical_feature]
            labels = [random.randint(0, 32) for _ in word_length]

        batch_idxes = FixedBucketSampler([len(word_f) for word_f in word_feature], batch_size, num_buckets=num_buckets)
        batch = []
        for batch_idx in batch_idxes:
            batch_features = [[] for _ in range(len(features))]
            batch_labels = []
            for idx in batch_idx:
                for i, feature in enumerate(batch_features):
                    batch_features[i].append(features[i][idx])
                batch_labels.append(labels[idx])
            batch_data = []
            word_mask = []
            char_mask = []
            for i, feature in enumerate(batch_features):
                max_len = max([len(fea) for fea in feature]) if not cnn_tag else cnn_len
                padder = PadSequence(max_len, pad_val=padding)
                feature, mask = zip(*[(padder(fea), len(fea)) for fea in feature])
                if i == 0:
                    word_mask = mask
                elif i == 2:
                    char_mask = mask
                batch_data.append(mx.nd.array(feature))
            batch_data.append(mx.nd.array(word_mask))
            batch_data.append(mx.nd.array(char_mask))
            batch_data.append(mx.nd.array(batch_labels, dtype=np.int))
            batch.append(batch_data)
        return batch[::-1]

    @staticmethod
    def sym_gen(
            # word_embedding_size=352183, word_radical_embedding_size=31759,
            # char_embedding_size=4746, char_radical_embedding_size=242,
            word_embedding_size=1000, word_radical_embedding_size=1000,
            char_embedding_size=1000, char_radical_embedding_size=1000,
    ):
        # 在这里定义网络结构
        return RLSTM_CNNATN_CROSS(
            word_embedding_size=word_embedding_size,
            word_radical_embedding_size=word_radical_embedding_size,
            char_embedding_size=char_embedding_size,
            char_radical_embedding_size=char_radical_embedding_size,
        ) if not cnn_tag else RLSTM_CNN3D(
            word_embedding_size=word_embedding_size,
            word_radical_embedding_size=word_radical_embedding_size,
            char_embedding_size=char_embedding_size,
            char_radical_embedding_size=char_radical_embedding_size,
        )

    # 以下部分定义训练相关的方法
    @staticmethod
    def parameters_stabilize(net, select='.*embedding'):
        """
        固定部分参数
        Parameters
        ----------
        net
        select
        Returns
        -------

        """
        params = net.collect_params(select)
        pers_keys = []
        for k in params:
            pers_keys.append(k)
        for k in pers_keys:
            params.get(k).grad_req = 'null'

    @staticmethod
    def save_params(filename, net, select='^(?!.*embedding)'):
        net.collect_params(select).save(filename)

    @staticmethod
    def net_initialize(net, model_ctx, initializer=mx.init.Xavier()):
        # 初始化网络参数
        net.collect_params().initialize(initializer, ctx=model_ctx)

    @staticmethod
    def get_trainer(
            net, optimizer='adagrad',
            optimizer_params={
                'learning_rate': LEARNING_RATE,  # 32 0.007 > 0.005
                # 'gamma1': 0.999,
                # 'lr_scheduler': mx.lr_scheduler.FactorScheduler(step=2250, factor=0.99, stop_factor_lr=0.014),
            },
            params_select='^(?!.*embedding)'
    ):
        # 把优化器安装到网络上
        trainer = gluon.Trainer(net.collect_params(params_select), optimizer, optimizer_params)
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
                    net=net,
                    train_data=train_data, batch_size=batch_size,
                    trainer=trainer, bp_loss_f=bp_loss_f,
                    loss_function=loss_function, losses_monitor=losses_monitor,
                    ctx=ctx,
                    batch_informer=informer,
                )
                if informer:
                    informer.batch_end(batch_num)

                train_time = epoch_timer.end(wall=True) if epoch_timer else None

                # todo 定义每一轮结束后的模型评估方法
                test_eval_res = RLSTMModule.eval(evaluator, test_data, net, ctx)
                print(evaluator.format_eval_res(epoch, test_eval_res, loss_values, train_time,
                                                logger=evaluator.logger, log_f=evaluator.log_f)[0])

                # todo 定义模型保存方案
                if kwargs.get('prefix'):
                    RLSTMModule.save_params(kwargs['prefix'] + "-%04d.parmas" % (epoch + 1), net)

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
            for i, (word, word_radical, char, char_radical, word_mask, char_mask, label) in enumerate(train_data):
                fit_f(
                    net=net, batch_size=batch_size,
                    word=word, word_radical=word_radical, char=char, char_radical=char_radical,
                    word_mask=word_mask, char_mask=char_mask,
                    label=label,
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
    def eval(evaluater, test_data, net, ctx=mx.cpu()):
        # 在这里定义数据评估方法
        from tqdm import tqdm
        for i, (word, word_radical, char, char_radical, word_mask, char_mask, label) in enumerate(
                tqdm(test_data, desc="evaluating")):
            from longling.lib.candylib import as_list
            ctx = as_list(ctx)[0]
            # word = gutil.split_and_load(word, ctx)
            # word_radical = gutil.split_and_load(word_radical, ctx)
            # char = gutil.split_and_load(char, ctx)
            # char_radical = gutil.split_and_load(char_radical, ctx)
            #
            # word_mask = gutil.split_and_load(word_mask, ctx)
            # char_mask = gutil.split_and_load(char_mask, ctx)
            #
            # label = gutil.split_and_load(label, ctx)

            word = word.as_in_context(ctx)
            word_radical = word_radical.as_in_context(ctx)
            char = char.as_in_context(ctx)
            char_radical = char_radical.as_in_context(ctx)

            word_mask = word_mask.as_in_context(ctx)
            char_mask = char_mask.as_in_context(ctx)

            label = label.as_in_context(ctx)

            net.set_network_unroll(len(word[0]), len(char[0]))
            output = net(word, word_radical, char, char_radical, word_mask, char_mask)
            predictions = mx.nd.argmax(output, axis=1)
            evaluater.metrics.update(preds=predictions, labels=label)
        return dict(evaluater.metrics.get_name_value())

    @staticmethod
    def _fit_f(net, batch_size,
               word, word_radical, char, char_radical, word_mask, char_mask,
               label,
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
        data: Iterable
            The data for train
        label: Iterable
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
        from longling.lib.candylib import as_list
        ctx = as_list(ctx)
        if len(word) < len(ctx):
            ctx = ctx[:1]
        # todo
        words = gutil.split_and_load(word, ctx, even_split=False)
        word_radicals = gutil.split_and_load(word_radical, ctx, even_split=False)
        chars = gutil.split_and_load(char, ctx, even_split=False)
        char_radicals = gutil.split_and_load(char_radical, ctx, even_split=False)

        word_masks = gutil.split_and_load(word_mask, ctx, even_split=False)
        char_masks = gutil.split_and_load(char_mask, ctx, even_split=False)

        labels = gutil.split_and_load(label, ctx, even_split=False)

        # word = word.as_in_context(ctx)
        # word_radical = word_radical.as_in_context(ctx)
        # char = char.as_in_context(ctx)
        # char_radical = char_radical.as_in_context(ctx)
        #
        # word_mask = word_mask.as_in_context(ctx)
        # char_mask = char_mask.as_in_context(ctx)
        #
        # label = label.as_in_context(ctx)

        bp_loss = None
        with autograd.record():
            for (word, word_radical, char, char_radical, word_mask, char_mask, label) in zip(words, word_radicals,
                                                                                             chars, char_radicals,
                                                                                             word_masks, char_masks,
                                                                                             labels):
                net.set_network_unroll(len(word[0]), len(char[0]))
                output = net(word, word_radical, char, char_radical, word_mask, char_mask)  # todo
                for name, func in loss_function.items():
                    loss = func(output, label)  # todo
                    if name in bp_loss_f:
                        bp_loss = loss
                    loss_value = nd.mean(loss).asscalar()
                    if losses_monitor:
                        losses_monitor.update(name, loss_value)
        assert bp_loss is not None
        bp_loss.backward()
        trainer.step(batch_size)


if __name__ == '__main__':
    train_RLSTM()