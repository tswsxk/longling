# coding: utf-8
# created by tongshiwei on 17-12-9

from __future__ import print_function

import json
import logging

from tqdm import tqdm

from longling.base import *
from longling.lib.utilog import config_logging
from longling.lib.stream import wf_open, wf_close, rf_open

LABEL_PREFIX = '__label__'

logger = config_logging(logger='fasttext', console_log_level=logging.INFO, propagate=False)


def cast_file_format(location_ins, location_fast=None, label_prefix=LABEL_PREFIX):
    '''
    将特征和标签修改为fasttext可用的格式
    :param location_ins: 输入文件
    :return: 修改格式后的文件
    '''
    if location_fast is None:
        location_fast = location_ins + '.fast'

    logger.info("location_fast %s", location_fast)
    logger.info("location_instance %s", location_ins)

    wf = wf_open(location_fast)
    with rf_open(location_ins) as fin:
        for line in tqdm(fin):
            data = json.loads(line, encoding='utf8')
            x = ' '.join(data['x'].split())
            z = label_prefix + tostr(data['z'])
            line = '%s %s' % (z, x)
            print(line, file=wf)

    wf_close(wf)

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