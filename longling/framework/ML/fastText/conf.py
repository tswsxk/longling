# coding: utf-8
# created by tongshiwei on 17-12-9

from __future__ import print_function

import json
import logging

from tqdm import tqdm

from longling.base import *
from longling.lib.utilog import config_logging
from longling.lib.stream import wf_open, wf_close

LABEL_PREFIX = '__label__'

logger = config_logging(logger='fasttext', console_log_level=logging.INFO, propagate=False)

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
