#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
import operator as op

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora_cites', 'Dataset string.')  # 储存图的文件名
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 训练模型的名字GCN
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('out_dim', 8, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
# flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
# flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
# flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


_LAYER_UIDS = {}

# 用于生成随机的初始权重
'''在layer类中放在layer.name+'_vars'中，整个scope又放在model.name的scope中  
    self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],name='weights_' + str(i))'''
def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)

# 把稀疏矩阵转化为元组
def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx

# 建立一个全零矩阵作为变量
def zeros(shape, name=None):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

# 按行归一化特征向量
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

## 产生D-1/2 (A+I) D-1/2 \
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # 产生D-1/2 A D-1/2
    return sparse_to_tuple(adj_normalized)


# 稀疏张量的相乘
def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


# 导入数据 graph
def loaddata(dataset_str):
    # 用一个字典储存graph 格式为{index: [index_of_neighbor_nodes]}
    graph = {}
    # 图储存的格式为：每行两个数字 第一个是index，第二个是它指向的邻结点
    with open(dataset_str, 'r') as f:
        i = 1
        line = f.readline()
        tmp = line.split()
        while line:
            # print(i)
            if op.eq(line, ''):
                break
            else:
                if int(tmp[0]) in graph:
                    graph[int(tmp[0])].append(int(tmp[1]))
                else:
                    graph[int(tmp[0])] = []
                    graph[int(tmp[0])].append(int(tmp[1]))
                i = i+1
                line = f.readline()
                tmp = line.split()
    print("the number of edgs in this graph is", '%04d' % (i), '\n')  # 输出图的边数
    return graph

# （稀疏）矩阵乘法
def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

#gcn_layer
# 给每一个卷积层命名
def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class GraphConvolution(object):
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        # layer.__init__
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        # # #
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                print("inputdim is",input_dim,"out_dim",output_dim)
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim], name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

    def __call__(self, inputs):
        #with tf.name_scope(self.name):
        #    if self.logging and not self.sparse_inputs:
        #        tf.summary.histogram(self.name + '/inputs', inputs)

        # outputs = self._call(inputs)
        x = inputs
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
        # 卷积
        supports = list()
        for i in range(len(self.support)):# 式子中的l
            #pre_sup = tf.matmul(x,weights[i])
            pre_sup = dot(x, self.vars['weights_' + str(i)],sparse=self.sparse_inputs)
            #support = tf.matmul(self.support[i],pre_sup)
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output=tf.add_n(supports)
        # bias
        if self.bias:
            output += self.vars['bias']
        final_output=self.act(output)
        #if self.logging:
        #    tf.summary.histogram(self.name + '/outputs', outputs)
        return final_output

#gcn
class GCN(object):
    def __init__(self, placeholders,input_dim, **kwargs):
        # 原代码中调用model.__init___
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.layers = []
        self.activations = []
        self.outputs = None
        self.loss = 0
        self.accuracy = 0
        self.opt_op = None
        # # #
        self.inputs = placeholders['features']
        self.input_dim = input_dim  # 特征的维度
        # self.input_dim = self.inputs.get_shape().as_list()[1] # 会报错
        # self.input_dim = placeholders['features'].get_shape().as_list()[1] # 会报错
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        #_建立两个卷积层
        with tf.variable_scope(self.name):
            self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                                output_dim=FLAGS.hidden1,
                                                placeholders=self.placeholders,
                                                act=tf.nn.relu,
                                                dropout=True,
                                                sparse_inputs=True,
                                                logging=self.logging))
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.out_dim,
                                                placeholders=self.placeholders,
                                                act=lambda x: x,
                                                dropout=True,
                                                logging=self.logging))

        # Build sequential layer model 两层卷积
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1] # 保存下来这个

        #没想到用什么loss function
        # loss = 0
        # opt_op = tf.train.AdamOptimizer.minimize(loss)

def main():
    # 用一个字典graph表示输入的图 格式为{index: [index_of_neighbor_nodes]} 并生成邻接矩阵adj
    graph = loaddata(FLAGS.dataset)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # 因没有特征向量 用单位矩阵作为特征向量 然后按行归一化特征向量
    feature= np.eye(len(adj.todense()))
    features = sp.csr_matrix(feature).tolil()
    features = preprocess_features(features)
    # 方便后面计算，将adj转换成DAD,用support表示
    support = [preprocess_adj(adj)]
    num_supports= 1 #len(support) 暂时不知道用途
    print (features[2])
    print (features[1].shape)
    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        # features[2]是features的shape
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)
        # 'out_dim': tf.placeholder(tf.int32),
    }
    x=tf.placeholder(tf.int32)
    # 构建模型
    model = GCN(placeholders,input_dim=features[2][1])

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print ("begin training")
    for epoch in range(FLAGS.epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
        feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({x:3})
        # Training step
        # outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        outs = sess.run(model.outputs,feed_dict=feed_dict)
        print(outs)
        # print(len(outs))
        with open("out_vector.txt","w") as fo:
            for i in range(len(outs)):
                fo.write(str(outs[i]))
                fo.write('\n')
    print ("success")
if __name__ ==  '__main__':
    main()
