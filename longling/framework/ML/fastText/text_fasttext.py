# coding:utf-8
'''
二分类fasttext
'''

import datetime
import time
import fasttext
import json
import os
import logging
from evaluate import evaluate

LABEL_PREFIX = '__label__'

logger = logging.getLogger('fasttext')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def cast_file_format(location_ins):
    '''
    将特征和标签修改为fasttext可用的格式
    :param location_ins: 输入文件
    :return: 修改格式后的文件
    '''
    location_fast = location_ins + '.fast'
    logger.info("location_fast %s", location_fast)
    logger.info("location_instance %s", location_ins)

    with open(location_fast, 'w') as fout:
        with open(location_ins) as fin:
            for line in fin:
                data = json.loads(line, encoding='utf8')
                x = ' '.join(data['x'].split())
                z = LABEL_PREFIX + str(data['z'])
                line = ('%s %s' % (z, x)).encode('utf8')
                print >> fout, line

    return location_fast


def get_parameters(paras):
    '''
    fasttext参数设定
    :param paras: 输入参数
    :return: 
    '''
    parameters = dict()
    parameters['lr'] = paras.get('lr', 0.1)
    parameters['lr_update_rate'] = paras.get('lr_update_rate', 100)
    parameters['dim'] = paras.get('dim', 60)
    parameters['ws'] = paras.get('ws', 3)
    parameters['min_count'] = paras.get('min_count', 5)
    parameters['silent'] = 0
    parameters['word_ngrams'] = paras.get('word_ngrams', 1)
    parameters['loss'] = paras.get('loss', 'softmax')
    parameters['bucket'] = paras.get('bucket', 0)
    parameters['t'] = paras.get('t', 0.0001)
    parameters['thread'] = paras.get('thread', 4)
    parameters['minn'] = paras.get('minn', 0)
    parameters['maxn'] = paras.get('maxn', 0)
    parameters['neg'] = paras.get('neg', 5)
    return parameters


def process_fit(location_ins, location_test, model_dir, prop={}, epoch=50):
    '''
    训练模型
    :param location_ins: 训练数据文件
    :param location_test: 测试数据文件
    :param model_dir: 模型目录
    :param prop: 参数
    :param epoch: 训练轮数
    :return: 
    '''
    location_log = os.path.join(model_dir, 'model.log')
    f_log = open(location_log, mode='w')
    logger.info("开始转换训练文件格式")
    # print "Start to cast file format"
    cast_tic = time.time()
    location_data = cast_file_format(location_ins)
    location_model = os.path.join(model_dir, 'model')
    logger.info("转换文件格式耗时 %ss" % (time.time() - cast_tic))
    # print "It takes %ss to cast_file" % (time.time() - cast_tic)
    tic = time.time()
    model = fasttext.supervised(
        location_data,
        location_model,
        label_prefix=LABEL_PREFIX,
        lr=prop.get('lr', 0.1),
        lr_update_rate=prop.get('lr_update_rate', 100),
        dim=prop.get('dim', 60),
        ws=prop.get('ws', 3),
        min_count=prop.get('min_count', 5),
        neg=prop.get('neg', 5),
        minn=prop.get('minn', 0),
        maxn=prop.get('maxn', 0),
        epoch=epoch,
        silent=0,
        word_ngrams=prop.get('word_ngrams', 1),
        loss=prop.get('loss', 'softmax'),
        bucket=prop.get('bucket', 0),
        t=prop.get('t', 0.0001),
        thread=prop.get('thread', 4)
    )
    logger.info("模型参数设置如下：")
    parameters = get_parameters(prop)
    parameters['epoch'] = epoch
    for k, v in parameters.items():
        logger.info("%s\t%s" % (k, v))

    train_time = time.time() - tic
    logger.info("训练开始于：%s" % datetime.datetime.time(datetime.datetime.now()))
    # print "finish training fasttext, it takes %ss" % train_time
    logger.info("训练Fasttext耗时%ss" % train_time)
    # print "test performance"

    classifier = fasttext.load_model(location_model+'.bin', label_prefix=LABEL_PREFIX)
    predicts = []
    golds = []
    texts = []
    with open(location_test) as fin:
        for line in fin:
            data = json.loads(line, encoding='utf8')
            x = ' '.join(data['x'].split())
            if len(data['x'].split()) == 0:
                continue
            z = LABEL_PREFIX + str(data['z'])
            texts.append(x)
            golds.append(str(data['z']))
            if len(texts) == 1000:
                labels = classifier.predict_label(texts, 1)
                texts = []
                for l in labels:
                    predicts.append(l[0])
    if len(texts) > 0:
        labels = classifier.predict_label(texts, 1)
        for l in labels:
            predicts.append(l[0])
    assert len(predicts) == len(golds)
    cat_res = evaluate(golds, predicts)

    num_correct = 0.
    num_total = 0.
    for i in xrange(len(predicts)):
        if predicts[i] == golds[i]:
            num_correct += 1.
        num_total += 1.
    dev_acc = num_correct * 100 / float(num_total)
    print >> f_log, '--- Dev Accuracy thus far: %.3f' % dev_acc
    print '--- Dev Accuracy thus far: %.3f' % dev_acc
    for cat, res in cat_res.items():
        print >> f_log, '--- Dev Category %s P=%s,R=%s,F=%s' % (cat, res[0], res[1], res[2])
        print '--- Dev Category %s P=%s,R=%s,F=%s' % (cat, res[0], res[1], res[2])

    f_log.close()

def process_batch_score(location_dat, location_res, location_model):
    print "save score results to", location_res
    print "pre-trained model in", location_model
    classifier = fasttext.load_model(location_model, label_prefix=LABEL_PREFIX)
    texts = []
    batch = []
    cat1, cat0 = '1', '0'
    with open(location_res, 'w') as fout:
        with open(location_dat) as fin:
            for line in fin:
                data = json.loads(line)
                xx = data['data'].split()
                if len(xx) == 0:
                    continue
                x = ' '.join(xx)

                batch.append(data)
                texts.append(x)
                if len(texts) == 1000:
                    labels = classifier.predict_proba(texts, 1)
                    for i, l in enumerate(labels):
                        cat, prob = l[0]
                        batch[i]['score'] = (prob if cat == cat1 else (1. - prob))
                        print >> fout, json.dumps(batch[i], ensure_ascii=False).encode('utf8')
                    texts = []
                    batch = []
            labels = classifier.predict_proba(texts, 1)
            for j in xrange(len(batch)):
                cat, prob = labels[j][0]
                batch[j]['score'] = (prob if cat == cat1 else (1. - prob))
                print >> fout, json.dumps(batch[j], ensure_ascii=False).encode('utf8')


