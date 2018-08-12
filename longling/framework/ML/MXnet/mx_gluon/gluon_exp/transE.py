# coding:utf-8
# created by tongshiwei on 2018/7/13
"""
对gluon的训练、测试过程进行封装
fit_f = epoch_loop(batch_loop(fit_f))
复制此文件以进行修改
大量使用 staticmethod 并用 get_params 对参数进行分离的原因是因为耦合性太高会导致改起来不太方便
可能修改的地方用 todo 标出
"""
import json
import os
import random

from collections import OrderedDict

from tqdm import tqdm

import mxnet as mx
from mxnet import autograd, nd
from mxnet import gluon

from longling.lib.utilog import config_logging, LogLevel
from longling.lib.clock import Clock
from longling.framework.ML.MXnet.mx_gluon.gluon_toolkit import TrainBatchInformer, Evaluator, MovingLosses
from longling.framework.ML.MXnet.viz import plot_network

from longling.lib.stream import wf_open
from longling.framework.ML.MXnet.mx_gluon.gluon_sym import TransE
from longling.framework.ML.MXnet.mx_gluon.gluon_toolkit import get_l2_embedding_weight
from longling.framework.ML.MXnet.mx_gluon.gluon_sym import PairwiseLoss


########################################################################
# write the user function here
def idx2str(map_file):
    with open(map_file) as f:
        mapping = json.load(f)
    mapping.update({'</s>': 0})
    return dict([(strid, idx) for idx, strid in mapping.items()])


def embedding2file(filename, embedding_weight, embedding_map):
    with wf_open(filename) as wf:
        for idx, weight in tqdm(enumerate(embedding_weight), filename):
            print("%s %s" % (embedding_map[idx], " ".join([str(float('%.6f' % value)) for value in weight])), file=wf)


def embedding_persistence(net, entities_embedding_path, relations_embedding_path, entities_map, relations_map,
                          entities_size, relations_size, batch_size=None):
    entity_embedding = get_l2_embedding_weight(net.entity_embedding, entities_size, batch_size=batch_size,
                                               prefix="entity")
    embedding2file(entities_embedding_path, entity_embedding, entities_map)
    relation_embedding = get_l2_embedding_weight(net.relation_embedding, relations_size, batch_size=batch_size,
                                                 prefix="relation")
    embedding2file(relations_embedding_path, relation_embedding, relations_map)


#########################################################################


# todo 重命名eval_transE函数到需要的模块名
def eval_transE():
    root = "../../../../"
    model_name = "transE"
    model_dir = root + "data/gluon/%s/" % model_name
    vec_dir = root + "data/KG/FB15/"
    entities_idx2str = "freebase_mtr100_mte100-entities.txt"
    relations_idx2str = "freebase_mtr100_mte100-relations.txt"
    entities_embedding = "entity.vec.dat"
    relations_embedding = "relation.vec.dat"
    big_test_file = vec_dir + "big_test.jsonxz"
    full_test_file = vec_dir + "full_test.jsonxz"
    epoch_num = 122
    mod = TransEModule(
        model_dir=model_dir,
        model_name=model_name,
        vec_dir=vec_dir,
        entities_idx2str=entities_idx2str,
        relations_idx2str=relations_idx2str,
        entities_embedding=entities_embedding,
        relations_embedding=relations_embedding,
        ctx=mx.cpu()
    )
    entities_idx2str = idx2str(mod.entities_idx2str)
    relations_idx2str = idx2str(mod.relations_idx2str)
    net = TransEModule.sym_gen(len(entities_idx2str), len(relations_idx2str), 50, 'L1')
    net.load_params(mod.prefix + "-%04d.parmas" % epoch_num, mod.ctx)
    test_data = TransEModule.get_full_test_iter(full_test_file)
    print(TransEModule.eval(test_data, net, mod.ctx))


# todo 重命名use_transE函数到需要的模块名
def use_transE():
    root = "../../../../../"
    model_name = "transE"
    model_dir = root + "data/gluon/%s/" % model_name
    vec_dir = root + "data/KG/FB15/"
    entities_idx2str = "freebase_mtr100_mte100-entities.txt"
    relations_idx2str = "freebase_mtr100_mte100-relations.txt"
    entities_embedding = "entity.vec.dat"
    relations_embedding = "relation.vec.dat"
    epoch_num = 122
    dim = 50
    d = 'L1'
    mod = TransEModule(
        model_dir=model_dir,
        model_name=model_name,
        vec_dir=vec_dir,
        entities_idx2str=entities_idx2str,
        relations_idx2str=relations_idx2str,
        entities_embedding=entities_embedding,
        relations_embedding=relations_embedding,
        ctx=mx.cpu()
    )
    entities_idx2str = idx2str(mod.entities_idx2str)
    relations_idx2str = idx2str(mod.relations_idx2str)
    net = TransEModule.sym_gen(len(entities_idx2str), len(relations_idx2str), dim, d)
    net = mod.load(net, epoch_num, mod.ctx)
    embedding_persistence(net, mod.entities_embedding, mod.relations_embedding, entities_idx2str, relations_idx2str,
                          len(entities_idx2str), len(relations_idx2str))


# todo 重命名train_transE函数到需要的模块名
def train_transE():
    # 1 配置参数初始化
    root = "../../../../"
    model_name = "transE"
    model_dir = root + "data/gluon/%s/" % model_name
    vec_dir = root + "data/KG/FB15/"
    entities_idx2str = "freebase_mtr100_mte100-entities.txt"
    relations_idx2str = "freebase_mtr100_mte100-relations.txt"
    entities_embedding = "entity.vec.dat"
    relations_embedding = "relation.vec.dat"
    train_file = vec_dir + "train.jsonxz"
    test_file = vec_dir + "test.jsonxz"
    big_test_file = vec_dir + "big_test.jsonxz"
    full_train_file = vec_dir + "freebase_mtr100_mte100-train"

    mod = TransEModule(
        model_dir=model_dir,
        model_name=model_name,
        vec_dir=vec_dir,
        entities_idx2str=entities_idx2str,
        relations_idx2str=relations_idx2str,
        entities_embedding=entities_embedding,
        relations_embedding=relations_embedding,
        ctx=mx.cpu()
    )
    logger = config_logging(logger=model_name, console_log_level=LogLevel.INFO)
    logger.info(str(mod))

    ############################################################################
    # experiment params
    dim = 50
    gamma = 1
    d = 'L1'
    lr = 0.01
    ############################################################################

    entities_idx2str = idx2str(mod.entities_idx2str)
    relations_idx2str = idx2str(mod.relations_idx2str)

    # 2 todo 定义网络结构并保存
    logger.info("generating symbol")
    net = TransEModule.sym_gen(len(entities_idx2str), len(relations_idx2str), dim, d)
    net.hybridize()

    # 5 todo 定义训练相关参数
    begin_epoch = 0
    epoch_num = 200
    epoch_step = 4
    batch_size = 128
    ctx = mod.ctx

    # 3 todo 自行设定网络输入，可视化检查网络
    logger.info("visualization")
    data_shape = (1,)
    viz_shape = {
        'sub': (batch_size,) + data_shape,
        'rel': (batch_size,) + data_shape,
        'obj': (batch_size,) + data_shape,
    }
    sub = mx.sym.var("sub")
    rel = mx.sym.var("rel")
    obj = mx.sym.var("obj")
    sym = net(sub, rel, obj)
    plot_network(
        nn_symbol=sym,
        save_path=model_dir + "plot/network",
        shape=viz_shape,
        node_attrs={"fixedsize": "false"},
        view=False
    )

    # 5 todo 定义损失函数
    # bp_loss_f 定义了用来进行 back propagation 的损失函数
    bp_loss_f = {"pairwise_loss": PairwiseLoss(None, -1, margin=gamma)}
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
    evaluator = Evaluator(
        # metrics=eval_metrics,
        model_ctx=mod.ctx,
        logger=validation_logger,
        log_f=mod.validation_result_file
    )

    # 4 todo 定义数据加载
    logger.info("loading data")
    # train_data = TransEModule.get_train_iter(train_file, batch_size=batch_size)
    train_data = TransEModule.get_full_random_train_iter(full_train_file, batch_size=batch_size,
                                                         entities_size=len(entities_idx2str),
                                                         relations_size=len(relations_idx2str))
    test_data = TransEModule.get_test_iter(test_file)

    # full_test_data = TransEModule.get_full_test_iter(full_test_file)

    # 6 todo 训练
    # 直接装载已有模型，确认这一步可以执行的话可以忽略 2 3 4
    logger.info("start training")
    try:
        net = mod.load(net, begin_epoch, mod.ctx)
        logger.info("load params from existing model file %s" % mod.prefix + "-%04d.parmas" % begin_epoch)
    except FileExistsError:
        logger.info("model doesn't exist, initializing")
        TransEModule.net_initialize(net, ctx)
    trainer = TransEModule.get_trainer(net, optimizer_params={'learning_rate': lr})
    mod.fit(
        net=net, begin_epoch=begin_epoch, epoch_num=epoch_step, batch_size=batch_size,
        train_data=train_data,
        trainer=trainer, bp_loss_f=bp_loss_f,
        loss_function=loss_function, losses_monitor=losses_monitor,
        test_data=test_data,
        ctx=ctx,
        informer=informer, epoch_timer=timer, evaluator=evaluator,
        prefix=mod.prefix,
    )
    net.export(mod.prefix)

    for epoch_idx in range(epoch_step, epoch_num, epoch_step):
        train_data = TransEModule.get_full_random_train_iter(full_train_file, batch_size=batch_size,
                                                             entities_size=len(entities_idx2str),
                                                             relations_size=len(relations_idx2str))
        mod.fit(
            net=net, begin_epoch=epoch_idx, epoch_num=epoch_idx + epoch_step,
            batch_size=batch_size,
            train_data=train_data,
            trainer=trainer, bp_loss_f=bp_loss_f,
            loss_function=loss_function, losses_monitor=losses_monitor,
            test_data=test_data,
            ctx=ctx,
            informer=informer, epoch_timer=timer, evaluator=evaluator,
            prefix=mod.prefix,
        )

    # optional todo 评估
    # big_test_data = TransEModule.get_test_iter(big_test_file)
    # TransEModule.eval(big_test_data, net, ctx, 10)

    # embedding persistence
    embedding_persistence(net, mod.entities_embedding, mod.relations_embedding, entities_idx2str, relations_idx2str,
                          len(entities_idx2str), len(relations_idx2str))


class TransEModule(object):
    """
    模块模板
    train 修改流程

    # 1
    修改 __init__ 和 params 方法
    TransEModule(....) 初始化一些通用的参数，比如模型的存储路径等

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

    def __init__(self, model_dir, model_name, vec_dir, entities_idx2str, relations_idx2str,
                 entities_embedding, relations_embedding, ctx=mx.cpu()):
        """

        Parameters
        ----------
        model_dir: str
            The directory to store the model and the corresponding files such as log
        model_name: str
            The name of this model
        vec_dir: str
            The directory to store the word-embedding file
        ctx: Context or list of Context
            Defaults to ``mx.cpu()``.
        """
        # 初始化一些通用的参数
        self.model_dir = os.path.abspath(model_dir)
        self.model_name = model_name
        self.vec_dir = os.path.abspath(vec_dir)
        self.entities_idx2str = os.path.join(self.vec_dir, entities_idx2str)
        self.relations_idx2str = os.path.join(self.vec_dir, relations_idx2str)
        self.entities_embedding = os.path.join(self.vec_dir, entities_embedding)
        self.relations_embedding = os.path.join(self.vec_dir, relations_embedding)
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
        params['vec_dir'] = self.vec_dir
        params['entities_idx2str'] = self.entities_idx2str
        params['relations_idx2str'] = self.relations_idx2str
        params['entities_embedding'] = self.entities_embedding
        params['relations_embedding'] = self.relations_embedding
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
    def get_train_iter(filename, batch_size, entities_map=None, relations_map=None):
        pos_subs, pos_rels, pos_objs = [], [], []
        neg_subs, neg_rels, neg_objs = [], [], []
        with open(filename) as f:
            for line in tqdm(f, desc="reading file[%s]" % filename):
                if not line.strip():
                    continue
                pos_neg = json.loads(line)
                pos_sub, pos_rel, pos_obj = pos_neg["x"]
                neg_sub, neg_rel, neg_obj = pos_neg["z"]
                if entities_map:
                    pos_sub = entities_map[pos_sub]
                    pos_obj = entities_map[pos_obj]
                    neg_sub = entities_map[neg_sub]
                    neg_obj = entities_map[neg_obj]
                else:
                    pos_sub = int(pos_sub)
                    pos_obj = int(pos_obj)
                    neg_sub = int(neg_sub)
                    neg_obj = int(neg_obj)
                if relations_map:
                    pos_rel = relations_map[pos_rel]
                    neg_rel = relations_map[neg_rel]
                else:
                    pos_rel = int(pos_rel)
                    neg_rel = int(neg_rel)
                pos_subs.append(pos_sub)
                pos_rels.append(pos_rel)
                pos_objs.append(pos_obj)

                neg_subs.append(neg_sub)
                neg_rels.append(neg_rel)
                neg_objs.append(neg_obj)

        return gluon.data.DataLoader(
            gluon.data.ArrayDataset(pos_subs, pos_rels, pos_objs, neg_subs, neg_rels, neg_objs), batch_size=batch_size,
            shuffle=True)

    @staticmethod
    def get_full_train_iter(filename, batch_size,
                            sro=None, ors=None, entities=None,
                            entities_map=None, relations_map=None):
        if sro and ors and entities:
            pass
        else:
            from longling.framework.KG.io_lib import rdf2sro, rdf2ors, load_plain
            sro, entities, _ = rdf2sro(load_plain(filename))
            ors, _, _ = rdf2ors(load_plain(filename))

        def re_gen():
            pos_subs, pos_rels, pos_objs = [], [], []
            neg_subs, neg_rels, neg_objs = [], [], []
            with open(filename) as f:
                for line in tqdm(f, desc="reading file[%s]" % filename):
                    if not line.strip():
                        continue
                    pos_sub, pos_rel, pos_obj = line.split()
                    sr_neg = entities - sro[pos_sub][pos_rel]
                    or_neg = entities - ors[pos_obj][pos_rel]
                    sr_len = len(sr_neg)
                    ro_len = len(or_neg)

                    neg_idx = random.randint(0, sr_len + ro_len)
                    neg_rel = pos_rel
                    if neg_idx < sr_len:
                        neg_obj = random.sample(sr_neg, 1)[0]
                        neg_sub = pos_sub
                    else:
                        neg_obj = random.sample(or_neg, 1)[0]
                        neg_sub = pos_sub

                    if entities_map:
                        pos_sub = entities_map[pos_sub]
                        pos_obj = entities_map[pos_obj]
                        neg_sub = entities_map[neg_sub]
                        neg_obj = entities_map[neg_obj]
                    else:
                        pos_sub = int(pos_sub)
                        pos_obj = int(pos_obj)
                        neg_sub = int(neg_sub)
                        neg_obj = int(neg_obj)
                    if relations_map:
                        pos_rel = relations_map[pos_rel]
                        neg_rel = relations_map[neg_rel]
                    else:
                        pos_rel = int(pos_rel)
                        neg_rel = int(neg_rel)
                    pos_subs.append(pos_sub)
                    pos_rels.append(pos_rel)
                    pos_objs.append(pos_obj)

                    neg_subs.append(neg_sub)
                    neg_rels.append(neg_rel)
                    neg_objs.append(neg_obj)
            yield gluon.data.DataLoader(
                gluon.data.ArrayDataset(pos_subs, pos_rels, pos_objs, neg_subs, neg_rels, neg_objs),
                batch_size=batch_size,
                shuffle=True)

        return list(re_gen())[0]

    @staticmethod
    def get_full_random_train_iter(filename, batch_size,
                                   entities_size, relations_size):
        """
        only support index file
        Parameters
        ----------
        filename
        batch_size
        entities_size
        relations_size

        Returns
        -------

        """

        pos_subs, pos_rels, pos_objs = [], [], []
        neg_subs, neg_rels, neg_objs = [], [], []
        with open(filename) as f:
            for line in tqdm(f, desc="reading file[%s]" % filename):
                if not line.strip():
                    continue
                pos_sub, pos_rel, pos_obj = line.strip().split()
                pos_sub = int(pos_sub)
                pos_rel = int(pos_rel)
                pos_obj = int(pos_obj)
                neg_idx = random.randint(0, 2)
                neg_rel = pos_rel
                if neg_idx < 1:
                    neg_obj = random.randint(0, entities_size)
                    neg_sub = pos_sub
                else:
                    neg_obj = random.randint(0, relations_size)
                    neg_sub = pos_sub

                pos_subs.append(pos_sub)
                pos_rels.append(pos_rel)
                pos_objs.append(pos_obj)

                neg_subs.append(neg_sub)
                neg_rels.append(neg_rel)
                neg_objs.append(neg_obj)
        return gluon.data.DataLoader(
            gluon.data.ArrayDataset(pos_subs, pos_rels, pos_objs, neg_subs, neg_rels, neg_objs),
            batch_size=batch_size,
            shuffle=True)

    @staticmethod
    def get_full_test_iter(filename, entities_map=None, relations_map=None, wrapper=lambda x, y: x):
        with open(filename) as f:
            for line in wrapper(f, "reading file[%s]" % filename):
                if not line.strip():
                    continue
                pos_neg = json.loads(line)
                pos_sub, pos_rel, pos_obj = pos_neg["x"]
                if entities_map:
                    pos_sub = entities_map[pos_sub]
                    pos_obj = entities_map[pos_obj]
                else:
                    pos_sub = int(pos_sub)
                    pos_obj = int(pos_obj)
                if relations_map:
                    pos_rel = relations_map[pos_rel]
                else:
                    pos_rel = int(pos_rel)
                negs = []
                for neg_triple in pos_neg["z"]:
                    neg_sub, neg_rel, neg_obj = neg_triple
                    if entities_map:
                        neg_sub = entities_map[neg_sub]
                        neg_rel = relations_map[neg_rel]
                        neg_obj = entities_map[neg_obj]
                    else:
                        neg_sub = int(neg_sub)
                        neg_obj = int(neg_obj)
                        neg_rel = int(neg_rel)
                    negs.append((neg_sub, neg_rel, neg_obj))
                yield [(pos_sub, pos_rel, pos_obj)] + negs

    @staticmethod
    def get_test_iter(filename, entities_map=None, relations_map=None):
        return list(TransEModule.get_full_test_iter(filename, entities_map, relations_map, wrapper=tqdm))

    @staticmethod
    def eval(test_data, net, model_ctx, top_n=10):
        # 在这里定义数据评估方法
        topn = 0
        for i, data in tqdm(enumerate(test_data), "testing"):
            subs, rels, objs = [], [], []
            for d in data:
                subs.append(d[0])
                rels.append(d[1])
                objs.append(d[2])
            eval_data = gluon.data.DataLoader(gluon.data.ArrayDataset(subs, rels, objs),
                                              batch_size=len(subs), shuffle=False)

            res = None
            for sub, rel, obj in eval_data:
                sub = sub.as_in_context(model_ctx)
                rel = sub.as_in_context(model_ctx)
                obj = sub.as_in_context(model_ctx)
                if res is None:
                    res = net(sub, rel, obj)
                else:
                    res.concat(net(sub, rel, obj))
            res = res.asnumpy().tolist()
            smaller = 0
            topn += 1
            for n in res[1:]:
                if n < res[0]:
                    smaller += 1
                if smaller >= top_n:
                    topn -= 1
                    break
        return {'hits@%s' % top_n: topn / i}

    @staticmethod
    def sym_gen(entities_size, relations_size, dim, d):
        # 在这里定义网络结构
        return TransE(entities_size, relations_size, dim, d)

    # 以下部分定义训练相关的方法
    @staticmethod
    def net_initialize(net, model_ctx, initializer=mx.init.Normal(sigma=.1)):
        # 初始化网络参数
        net.collect_params().initialize(initializer, ctx=model_ctx)

    @staticmethod
    def get_trainer(net, optimizer='sgd', optimizer_params={'learning_rate': .01}):
        # 把优化器安装到网络上
        trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
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
        begin_epoch:
            The begin epoch of this train procession
        epoch_num:
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
            begin_epoch:
                The begin epoch of this train procession
            epoch_num:
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
                    train_data=train_data,
                    batch_size=batch_size,
                    trainer=trainer, bp_loss_f=bp_loss_f,
                    loss_function=loss_function, losses_monitor=losses_monitor,
                    ctx=ctx,
                    batch_informer=informer,
                )
                if informer:
                    informer.batch_end(batch_num)

                train_time = epoch_timer.end(wall=True) if epoch_timer else None

                # todo 定义每一轮结束后的模型评估方法
                test_eval_res = TransEModule.eval(test_data, net, ctx, 3)
                print(evaluator.format_eval_res(epoch, test_eval_res, loss_values, train_time,
                                                logger=evaluator.logger, log_f=evaluator.log_f)[0])

                # todo 定义模型保存方案
                if kwargs.get('prefix'):
                    net.save_params(kwargs['prefix'] + "-%04d.parmas" % (epoch + 1))

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
                net,
                train_data,
                batch_size,
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
            for i, (pos_sub, pos_rel, pos_obj, neg_sub, neg_rel, neg_obj) in enumerate(train_data):
                fit_f(
                    net=net,
                    pos_sub=pos_sub, pos_rel=pos_rel, pos_obj=pos_obj,
                    neg_sub=neg_sub, neg_rel=neg_rel, neg_obj=neg_obj,
                    batch_size=batch_size,
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
    def _fit_f(net, batch_size,
               pos_sub, pos_rel, pos_obj, neg_sub, neg_rel, neg_obj,
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
        pos_sub: Iterable
            The positive subject for train
        pos_rel: Iterable
            The positive relation for train
        pos_obj: Iterable
            The positive object for train
        neg_sub: Iterable
            The negative subject for train
        neg_rel: Iterable
            The negative relation for train
        neg_obj: Iterable
            The negative object for train
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

        # todo
        pos_sub = pos_sub.as_in_context(ctx)
        pos_rel = pos_rel.as_in_context(ctx)
        pos_obj = pos_obj.as_in_context(ctx)
        neg_sub = neg_sub.as_in_context(ctx)
        neg_rel = neg_rel.as_in_context(ctx)
        neg_obj = neg_obj.as_in_context(ctx)

        bp_loss = None
        with autograd.record():
            neg_out = net(pos_sub, pos_rel, pos_obj)
            pos_out = net(neg_sub, neg_rel, neg_obj)
            for name, func in loss_function.items():
                loss = func(pos_out, neg_out)  # todo
                if name in bp_loss_f:
                    bp_loss = loss
                loss_value = nd.mean(loss).asscalar()
                if losses_monitor:
                    losses_monitor.update(name, loss_value)

        assert bp_loss is not None
        bp_loss.backward()
        trainer.step(batch_size)


if __name__ == '__main__':
    # train_transE()
    eval_transE()
    # use_transE()
