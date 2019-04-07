# coding: utf-8

from elasticsearch import Elasticsearch
from elasticsearch import helpers

import json

import logging
logger = logging.getLogger('es_updater')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARN)
formatter = logging.Formatter('%(name)s: %(asctime)s, %(levelname)s %(message)s')


class updateStore(object):
    '''
    通用的es接口
    '''
    def __init__(self,prop):
        self.es = Elasticsearch(hosts=[{'host':prop['HOST'],'port':prop['PORT']}])
        self.index = prop['INDEX']
        self.type = prop['TYPE']

    def update(self, _id, data):
        res = self.es.update(index=self.index,doc_type=self.type, id=_id, body={'doc': data})
        return res

    def delete_id(self, _id):
        return self.es.delete(index=self.index,doc_type =self.type, id=_id)

    def insert_from_file(self, filename, doc_conv=None, batch_size=2048):
        w_buffer = []
        cnt = 0
        with open(filename) as f:
            for line in f:
                line = json.loads(line)
                if doc_conv is not None:
                    line = doc_conv(line)
                w_buffer.append(line)
                if len(w_buffer) == batch_size:
                    logging.info("insert %s", cnt)
                    self.insert_batch(w_buffer)
                    w_buffer = []

        if w_buffer:
            logging.info("insert %s", cnt)
            self.insert_batch(w_buffer)

        logging.info("finished")


    def insert_batch(self, records):
        def gen_actions(records):
            for record in records:
                _id = record.get('_id', None)
                del(record['_id'])
                actions = {
                    '_index': self.index,
                    '_type': self.type,
                    '_id': _id,
                    '_source': record,
                }
                yield actions

        actions = gen_actions(records)
        _, errors = helpers.bulk(self.es, actions)

        if len(errors) > 0:
            raise ValueError(u'数据写入错误：' + str('\n'.join(errors)))