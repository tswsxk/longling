#coding:utf-8

from elasticsearch import Elasticsearch


default_S = {
        'settings': {
            'index': {
                'number_of_shards': 8,
                'number_of_replicas': 1
            }
        }
    }


class esStore(object):
    '''
    通用的es接口
    '''
    def __init__(self,prop):
        self.es = Elasticsearch(hosts=[{'host':prop['HOST'],'port':prop['PORT']}])

    def new_index(self, index, doc_type, map_schema, shards=default_S):
        self.es.indices.create(index=index, body=shards)
        self.es.indices.put_mapping(index=index, doc_type=doc_type, body=map_schema)

    def exists_index(self, index):
        return self.es.indices.exists(index)

    def remove_index(self, index):
        self.es.indices.delete(index=index)


if __name__ == '__main__':
    _DOC_SCHEME = {
        'text': {  # 长度限制为30字，列表展示用
            'type': 'string',
            'index': 'not_analyzed',
        },
        'uri': {
            'type': 'string',
            'index': 'no',
        },
        'label': {
            'type': 'string',
            'index': 'not_analyzed'
        },
        'data': {
            'type': 'string',
            'index': 'not_analyzed',
        },
        'score': {
            'type': 'float',
            'index': 'not_analyzed'
        },
        'batch': {
            'type': 'string',
            'index': 'not_analyzed',
        },
        'insert_time': {
            'type': 'date',
            'format': 'YYYY-MM-dd HH:mm:ss',
        },
        'revise_time': {
            'type': 'date',
            'format': 'YYYY-MM-dd HH:mm:ss',
        },
        'extra': {
            'type': 'string',  # json
            'index': 'no',
        },
        'info': {
            'type': 'string',  #
            'analyzer': 'whitespace',
        },
        'id': {
            'type': 'string',
            'index': 'not_analyzed',
        },
        'gid': {
            'type': 'string',
            'index': 'not_analyzed'
        },
        'dataset': {
            'type': 'string',
            'index': 'not_analyzed'
        }
    }

    map_schema = {'properties': _DOC_SCHEME}
    print 'c'

