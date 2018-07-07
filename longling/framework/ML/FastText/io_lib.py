# coding:utf-8
# created by tongshiwei on 2018/7/6

from __future__ import absolute_import

import json

from tqdm import tqdm

from .conf import LABEL_PREFIX, logger

from longling.base import tostr
from longling.lib.stream import wf_open, wf_close


def jsonxz2fast(source_jsonxz, target_fast=None, label_prefix=LABEL_PREFIX, data_key='x', label_key='z'):
    '''
    将jsonxz文件转为fasttext可用于训练的fast文件
    jsonxz文件格式
        ['x':"word1 word2 ... wordn", "z": %label%]
        ['x':"word1 word2 ... wordn", "z": %label%]
        ['x':"word1 word2 ... wordn", "z": %label%]
    'x' 对应的是 空格分割的分词后的句子
    :param source_jsonxz: 输入文件
    :param target_fast:
    :param label_prefix
    :return: 修改格式后的文件
    '''
    if target_fast is None:
        target_fast = source_jsonxz + '.fast'

    logger.info("file for model %s", target_fast)
    logger.info("source file %s", source_jsonxz)

    wf = wf_open(target_fast)
    with open(source_jsonxz) as f:
        for line in tqdm(f):
            data = json.loads(line, encoding='utf8')
            d = ' '.join(data[data_key])
            label = label_prefix + tostr(data[label_key])
            line = '%s %s' % (label, d)
            print(line, file=wf)

    wf_close(wf)

    return target_fast
