# coding: utf-8
# create by tongshiwei on 2019/5/25

import json

import pandas as pd


class Table(object):
    """
    表格类，包括两个主元素

    * header：表头
    * body：表格内容

    提供几个便捷的转换方法：
    * merge_table_header: 交换列

    提供两种持久化方法
    * dump2csv：把 header 和 body 以 csv 格式进行存储
    * dump：把 header 和 body 分开 存储到不同文件中，
        header 以 json 字典格式存储，内容为 index -> feature_index 的映射
        body 以 csv 格式存储
    """

    def __init__(self, iterable, header):
        self.header = header
        self.body = iterable

    def __iter__(self):
        return iter(self.body)

    def __next__(self):
        return next(self.body)

    # ########################### Property ################################
    @property
    def header2index(self):
        return {name: i for i, name in enumerate(self.header)}

    @property
    def index2header(self):
        return self.header

    def reset(self, iterable):
        self.body = iterable

    # ##################### Methods ###################################
    @staticmethod
    def merge_table_header(table, new_header):
        """切换表格表头"""
        assert set(table.header) == set(new_header)
        if list(table.header) == list(new_header):
            return
        new_header2index = {name: i for i, name in enumerate(new_header)}
        index2index = {
            i: new_header2index[name] for i, name in enumerate(table.header)
        }
        table.header = new_header

        def get_new_body():
            for elems in table:
                _elems = [None] * len(elems)
                for i, _elem in enumerate(elems):
                    _elems[index2index[i]] = _elem
                yield _elems

        table.body = iter(get_new_body())

    @staticmethod
    def load4pandas(dataframe):
        header = dataframe.columns

        def get_body():
            for v in dataframe.values:
                yield v.tolist()

        body = get_body()

        return header, body

    def to_pandas(self):
        return pd.DataFrame([row for row in self.body], columns=self.header)

    # ########################### Persistence #############################
    def dump2csv(self, fp, values_wrapper=lambda values: ",".join(values)):
        print(values_wrapper(self.header), file=fp)
        for elems in self:
            print(values_wrapper(elems), file=fp)

    def dump(self, body_fp, meta_fp,
             values_wrapper=lambda values: ",".join(values)):
        json.dump(
            {i: name for i, name in enumerate(self.header)}, meta_fp,
            indent=2, sort_keys=True
        )
        for elems in self:
            print(values_wrapper(elems), file=body_fp)
