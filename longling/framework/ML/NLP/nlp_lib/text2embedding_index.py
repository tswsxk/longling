# coding:utf-8
# created by tongshiwei on 2018/7/25

from longling.framework.ML.NLP import load_tup
from longling.framework.ML.universe.domain_converter.mapper import JsonxzWordsMapper, Mapper


# 1 建立词向量字典
# 如果有其它格式的词向量文件, 先统一成tup或者dat格式，后面都用tup格式为例, tup和dat格式可以用vec_converter.py里的方法来转换
def get_dict(loc_tup):
    return load_tup(loc_tup)


# 2 向词向量字典中插入index同时设定默认值, 生成转换用的mapper
def get_mapper(embedding_dict):
    default_key = '</s>'
    default_value = [0] * len(next(embedding_dict.values()))
    key_index, index_value = JsonxzWordsMapper.insert_index(embedding_dict, (default_key, default_value))
    ki_mapper = JsonxzWordsMapper(key_index)
    iv_mapper = Mapper(index_value)
    return ki_mapper, iv_mapper


# 3 将原有文本文件转换为index文件
def key2index(ki_mapper, source, target):
    """

    Parameters
    ----------
    ki_mapper: JsonxzWordsMapper
    source
    target

    Returns
    -------

    """
    ki_mapper.transform_f(source, target)


# 4 其它一些处理，merge或其它

# 5 把index数据转换为embedding数据
def index2embedding(iv_mapper, index_datas):
    """

    Parameters
    ----------
    iv_mapper: Mapper
    index_datas: iterable

    Returns
    -------

    """
    return iv_mapper.transform_b(index_datas)



