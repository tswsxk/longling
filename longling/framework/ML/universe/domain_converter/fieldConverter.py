# coding: utf-8
# created by tongshiwei on 17-11-8

import logging

from abc import ABCMeta, abstractmethod
import json

from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

from longling.lib.stream import wf_open, wf_close

from longling.lib.text_lib.word2vec import gen_w2v_file


class baseFieldConverter(object):
    __metaclass__ = ABCMeta

    def __init__(self, model_file):
        self.model = self.load_model(model_file)

    @abstractmethod
    def load_model(self, model_file):
        return None

    @abstractmethod
    def convert(self, data):
        return data

    @staticmethod
    def build_model(source, model_file, **kwargs):
        pass


class any2indexConverter(baseFieldConverter):
    def load_model(self, model_file):
        model = {}
        with open(model_file) as f:
            for line in f:
                if line.strip():
                    key, value = json.loads(line)
                    model[key] = value
        return model

    def convert(self, data):
        try:
            return self.model[data]
        except:
            logging.error("can't find %s" % data)
            return -1

    @staticmethod
    def build_model(dbBatchIter, model_file, **kwargs):
        doamin_container = set()
        for datas in tqdm(dbBatchIter):
            for data in datas:
                if len(data) == 1:
                    doamin_container.add(data[0])
                elif len(data) > 1:
                    doamin_container.add(tuple(data))
                else:
                    raise Exception("null")
        wf = wf_open(model_file)
        for i, item in tqdm(enumerate(list(doamin_container))):
            print >> wf, json.dumps([item, i])

        wf_close(wf)


class word2vecConverter(baseFieldConverter):
    def __init__(self, model_file, dim=100):
        baseFieldConverter.__init__(self, model_file=model_file)
        self.dim = dim

    def convert(self, data):
        try:
            return self.model[data].tolist()
        except:
            return [0.0] * self.dim

    def load_model(self, model_file):
        return Word2Vec.load(model_file)

    @staticmethod
    def build_model(input_filename, model_file, **kwargs):
        gen_w2v_file(input_filename, model_file, **kwargs)
