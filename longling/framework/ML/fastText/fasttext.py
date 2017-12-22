# coding: utf-8
# created by tongshiwei on 17-12-9

from __future__ import absolute_import

import os

import fasttext

from longling.lib.stream import checkDir

from .conf import get_parameters, cast_file_format, LABEL_PREFIX


def get_location_model(model_dir):
    location_model = os.path.join(model_dir, 'model')
    checkDir(model_dir)
    return location_model


class Fasttext(object):
    def __init__(self, **kwargs):
        self.parameters = get_parameters(kwargs)
        self.label_prefix = kwargs.get('label_prefix', LABEL_PREFIX)
        self.model = None

    def fit(self, train_dataset, model_dir, cast_file_tag=True, epoch=50):
        location_data = self.cast_file(train_dataset, cast_file_tag)
        location_model = get_location_model(model_dir)

        self.model = fasttext.supervised(
            location_data,
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

    def cast_file(self, filename, cast_file_tag, **kwargs):
        if cast_file_tag:
            return cast_file_format(filename, label_prefix=self.label_prefix, **kwargs)
        else:
            return filename

    @staticmethod
    def load_model(location_model, label_prefix=LABEL_PREFIX):
        model = fasttext.load_model(location_model + '.bin', label_prefix=label_prefix)
        fasttext_model = Fasttext()
        fasttext_model.model = model
        return fasttext_model

    @staticmethod
    def cast_file_format(filename, casted_filaname=None, label_prefix=LABEL_PREFIX):
        return cast_file_format(filename, casted_filaname, label_prefix)
