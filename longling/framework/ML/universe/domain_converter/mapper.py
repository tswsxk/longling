# coding:utf-8
# created by tongshiwei on 2018/7/22
"""
映射类文件
包括两个功能
    第一个功能是从数据中生成可用的mapper index：token -> idx
        额外的，可以向已有的key->value映射插入index，生成两个新的字典，这一方法在文本embedding中较常使用
    第二个功能是将给定的token数据转换为idx数据：[word0,...,wordn] -> [idx0,...idxn]
"""
import json
from collections import defaultdict

from tqdm import tqdm

from longling.lib.candylib import as_list
from longling.lib.stream import wf_open, wf_close


class Mapper(object):
    def __init__(self, map_dict=None, map_size=None, dict_sources=None):
        if map_dict:
            if not map_size:
                map_size = len(map_dict)
            self.map_dict = map_dict
            self.map_size = map_size
        else:
            assert dict_sources, "when map_dict is None or map_size is None, dict_sources should be specified"
            map_dict, map_size = self.get_dict(as_list(dict_sources))
            self.map_dict = map_dict
            self.map_size = map_size
        self.restore_dict = None

    def __getitem__(self, item):
        return self.map_dict[item]

    @staticmethod
    def get_dict(dict_sources, line_parse=lambda x: x):
        map_dict, idx = defaultdict(), 1
        for dict_source in dict_sources:
            with open(dict_source) as f:
                for line in tqdm(f, desc="build dict from file[%s]" % dict_sources):
                    line = line.strip()
                    if not line:
                        continue
                    parsed_datas = as_list(line_parse(line))
                    for parsed_data in parsed_datas:
                        map_dict[parsed_data] = idx
                        idx += 1
        return map_dict, idx

    @staticmethod
    def insert_index(source_datas, defualt_key_value=None):
        """
        向字典中插入index，把原字典拆分成两个新的字典
        分别对应两种映射
            key -> index
            index -> value
        在 default_key_value 给定时
            index 0 为插入点

        Parameters
        ----------
        source_datas: dict or list or tuple
        defualt_key_value: tuple

        Returns
        -------

        >>> defualt_key_value = ('</s>', [0, 0, 0])
        >>> source_datas = [
        ...     ('a', [1, 1, 1]),
        ...     ('b', [2, 2, 3]),
        ... ]
        >>> key_index, index_value = Mapper.insert_index(source_datas)
        >>> mapper = Mapper(key_index)
        >>> mapper['a']
        0
        >>> source_datas = {
        ...    'a': [1, 1, 1],
        ...    'b': [2, 2, 3],
        ... }
        >>> key_index, index_value = Mapper.insert_index(source_datas, defualt_key_value)
        >>> key_index['</s>']
        0
        >>> mapper = Mapper(key_index)
        >>> mapper['c']
        0
        >>> mapper['a']
        1
        """
        index_value = dict()
        offset = 0
        if defualt_key_value is not None:
            key_index = defaultdict(int)
            defualt_key, defualt_value = defualt_key_value
            key_index[defualt_key] = 0
            index_value[0] = defualt_value
            if defualt_key in source_datas:
                del source_datas[defualt_key]
            offset += 1
        else:
            key_index = dict()

        keys_values = source_datas.items() if isinstance(source_datas, dict) else source_datas
        for i, (key, value) in enumerate(keys_values):
            index = offset + i
            key_index[key] = offset + i
            index_value[index] = value

        return key_index, index_value

    def transform(self, elems):
        """
        定义元素如何进行token -> idx的转换
        Parameters
        ----------
        elems

        Returns
        -------

        """
        return [self.map_dict[elem] for elem in elems]

    def restore(self, elems):
        """
        定义元素如何进行idx -> token的转换
        Parameters
        ----------
        elems

        Returns
        -------

        """
        return [self.restore_dict[elem] for elem in elems]

    def get_restore_dict(self):
        self.restore_dict = dict()
        for k, v in self.map_dict.items():
            if v not in self.restore_dict:
                self.restore_dict[v] = k

    def transform_b(self, datas):
        for data in datas:
            yield self.transform(data)

    def restore_b(self, datas):
        if self.restore_dict is None:
            self.get_restore_dict()
        for data in datas:
            yield self.restore(data)

    def transform_f(self, source, target):
        """

        Parameters
        ----------
        source: str
        target: str

        Returns
        -------

        """
        wf = wf_open(target)
        with open(source) as f:
            for line in tqdm(f, "%s transform %s to %s" % (self.__class__.__name__, source, target)):
                line = line.strip()
                if line:
                    print(self.transform(line), file=wf)
        wf_close(wf)

    def restore_f(self, source, target):
        """
        将 映射后的文件恢复 为 token文件
        执行一次 transform & restore 以后，可以清除所有不在mapper映射字典中的变量
        Parameters
        ----------
        source
        target

        Returns
        -------

        """
        wf = wf_open(target)
        if self.restore_dict is None:
            self.get_restore_dict()
        with open(source) as f:
            for line in tqdm(f, "%s restore %s to %s" % (self.__class__.__name__, source, target)):
                line = line.strip()
                if line:
                    print(self.restore(line), file=wf)
        wf_close(wf)

    def save(self, unite_map_filename):
        wf = wf_open(unite_map_filename)
        json.dump(self.map_dict, wf, ensure_ascii=False)
        wf_close(wf)

    @staticmethod
    def load(dict_map_filename):
        with open(dict_map_filename) as f:
            dict_map = defaultdict(int)
            dict_map.update(json.load(f))
            dict_size = len(dict_map) + 1
        return dict_map, dict_size


class JsonxzWordsMapper(Mapper):
    """
    >>> defualt_key_value = ('</s>', [0, 0, 0])
    >>> source_datas = {
    ...    'a': [1, 1, 1],
    ...    'b': [2, 2, 3],
    ... }
    >>> key_index, index_value = Mapper.insert_index(source_datas, defualt_key_value)
    >>> mapper = JsonxzWordsMapper(key_index)
    >>> test_datas = [
    ...    '{"x": ["a", "b", "c"], "z": 1}',
    ...    '{"x": ["b", "b", "c"], "z": 1}',
    ...    '{"x": ["c", "c"], "z": 1}'
    ... ]
    >>> test_datas = mapper.transform_b(test_datas)
    >>> test_datas = list(test_datas)
    >>> test_datas
    ['{"x": [1, 2, 0], "z": 1}', '{"x": [2, 2, 0], "z": 1}', '{"x": [0, 0], "z": 1}']
    >>> test_datas = mapper.restore_b(test_datas)
    >>> list(test_datas)
    ['{"x": ["a", "b", "</s>"], "z": 1}', '{"x": ["b", "b", "</s>"], "z": 1}', '{"x": ["</s>", "</s>"], "z": 1}']
    """

    def transform(self, line):
        """

        Parameters
        ----------
        line: str

        Returns
        -------

        """
        xz = json.loads(line)
        x = [self.map_dict[x] for x in xz['x']]
        return json.dumps({'x': x, 'z': xz['z']}, ensure_ascii=False)

    def restore(self, line):
        """

        Parameters
        ----------
        line

        Returns
        -------

        """
        xz = json.loads(line)
        x = [self.restore_dict[x] for x in xz['x']]
        return json.dumps({'x': x, 'z': xz['z']}, ensure_ascii=False)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
