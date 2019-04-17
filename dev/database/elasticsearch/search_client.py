# coding: utf-8
from elasticsearch import Elasticsearch
from elasticsearch import helpers
# import helpers

class searchStore(object):
    '''
    通用的es接口
    '''
    def __init__(self, prop):
        self.es = Elasticsearch(hosts=[{'host': prop['HOST'], 'port':prop['PORT']}])
        self.index = prop['INDEX']
        self.type = prop['TYPE']

    def get(self, i, fields='*'):
        index_name = self.index
        res = self.es.get(index=index_name,doc_type=self.type,id=i,_source_include=fields)
        return res['_source']

    def get_all_docs(self, fields='*'):
        q1 = {"match_all": {}}
        q = {"query": {
            "bool": {
                "must": [q1]
            }
        }}
        print q, self.es, self.index, self.type
        res = helpers.scan(client=self.es, index=self.index, doc_type=self.type,
                           query=q, scroll='10m', search_type='scan', \
                           _source_include=fields, size=500)
        return res
    def get_docs(self, st, et, fields='*'):
        T1 = st.strftime('%Y-%m-%dT%H:%M:%S')
        T2 = et.strftime('%Y-%m-%dT%H:%M:%S')
        q1 = {'range':{
            'create_time':{
                "gte": T1,
                "lt": T2
            }
        }}
        q = {"query":{
            "bool": {
                "must":[q1]
            }
        }}

        res = helpers.scan(client=self.es, index=self.index,doc_type=self.type, \
                                query=q,scroll='10m',search_type='scan',\
                                _source_include=fields,size=500)

        return res

def get_search_store(host, port, index, type):
    prop = {
        'HOST': host,
        'PORT': port,
        'INDEX': index,
        'TYPE': type,
    }
    return searchStore(prop)

if __name__ == '__main__':
    searchStore = get_search_store('10.3.132.206', 9205, "comment_text", "comment_text")
    datas = searchStore.get_all_docs()
    for data in datas:
        print data