# coding: utf-8
# created by tongshiwei on 17-12-9

from __future__ import absolute_import

import os

import fasttext

from longling.lib.stream import build_dir

from .conf import get_parameters, LABEL_PREFIX


def get_location_model(model_dir):
    location_model = os.path.join(model_dir, 'model')
    build_dir(model_dir)
    return location_model


class Fasttext(object):
    def __init__(self, **kwargs):
        self.parameters = get_parameters(kwargs)
        self.label_prefix = kwargs.get('label_prefix', LABEL_PREFIX)
        self.model = None

    def fit(self, train_file, model_dir, epoch=50, cast_file=None):
        location_fast = cast_file(train_file) if cast_file else train_file
        location_model = get_location_model(model_dir)

        self.model = fasttext.supervised(
            location_fast,
            location_model,
            label_prefix=LABEL_PREFIX,
            lr=self.parameters.get('lr', 0.1),
            lr_update_rate=self.parameters.get('lr_update_rate', 100),
            dim=self.parameters.get('dim', 60),
            ws=self.parameters.get('ws', 3),
            min_count=self.parameters.get('min_count', 5),
            neg=self.parameters.get('neg', 5),
            minn=self.parameters.get('minn', 0),
            maxn=self.parameters.get('maxn', 0),
            epoch=self.parameters.get('epoch', epoch),
            silent=self.parameters.get('silent', 0),
            word_ngrams=self.parameters.get('word_ngrams', 1),
            loss=self.parameters.get('loss', 'softmax'),
            bucket=self.parameters.get('bucket', 0),
            t=self.parameters.get('t', 0.0001),
            thread=self.parameters.get('thread', 4)
        )

        return self.model, location_model

    def predict(self, datas, kbest=1):
        '''
        预测k个最可能的label
        :param datas:
        :param kbest:
        :return:
        '''
        assert self.model is not None
        return self.model.predict_label(datas, kbest)

    def predict_probs(self, datas, label='1', kbest=1):
        assert self.model is not None
        predict_probs = []
        res = self.model.predict_probs(datas, kbest)[0][0]
        for l_p in res:
            l, p = l_p
            p = p if l == label else (1. - p)
            predict_probs.append(p)

    @staticmethod
    def load_model(location_model, label_prefix=LABEL_PREFIX):
        '''
        装载已有的fasttext模型
        :param location_model: 模型位置
        :param label_prefix:
        :return:
        '''
        model = fasttext.load_model(location_model + '.bin', label_prefix=label_prefix)
        fasttext_model = Fasttext()
        fasttext_model.model = model
        return fasttext_model

