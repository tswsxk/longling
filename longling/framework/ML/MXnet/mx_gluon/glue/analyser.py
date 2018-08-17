# coding: utf-8
# create by tongshiwei on 2018/8/16
"""
此模块用来针对result.json中的数据进行数据分析
开发测试中，非稳定版本
"""
import math
import matplotlib.pyplot as plt
import re
from collections import defaultdict


class ResultAnalyser(object):
    """
    Examples
    --------
    >>> result_analyser = ResultAnalyser()
    >>> dict_data = {'key1': {'key2': [1, 2, 3], 'key3': [1, 2]}}
    >>> result_analyser.add_record(dict_data)
    >>> 'key1_key2' in result_analyser.records
    True
    >>> 'key1_key3' in result_analyser
    True
    """

    def __init__(self):
        self.records = defaultdict(list)

    def add_record(self, dict_data, prefix=''):
        """
        添加字典数据并解析，如果字典的value也是字典的话，会深入解析，键名会进行拼接操作，如: {'key1': {'key2': value}}
        会返回 (key1_key2, value)

        Parameters
        ----------
        dict_data: dict
            字典格式数据
        prefix: str
        """
        for key, value in dict_data.items():
            if isinstance(value, dict):
                self.add_record(value, prefix=prefix + '_' + str(key) if prefix else str(key))
            else:
                self.records[prefix + '_' + str(key) if prefix else str(key)].append(value)

    def __iter__(self):
        return iter(self.records)

    def keys(self):
        return self.records.keys()

    def items(self):
        return self.records.items()

    def values(self):
        return self.records.values()

    def select(self, select='iteration'):
        pattern = re.compile(select)
        key_value = []
        for key, value in self.items():
            if pattern.match(key):
                key_value.append((key, value))
        return key_value

    def selects(self, selects=('accuracy|prf_avg_*', 'prf_\d+_.*',)):
        patterns = [re.compile(select) for select in selects]
        data = [[] for _ in patterns]
        for key, value in self:
            for i, pattern in enumerate(patterns):
                if pattern.match(key):
                    data[i].append((key, value))
        return data

    def visual_selects(self, x_select='iteration', y_selects=('accuracy|prf_avg_.*', 'prf_\d+_f1')):
        """
        按 x，y 格式选取数据，x_select 和 y_selects 都是正则匹配式，采用 re.match 方式匹配

        Parameters
        ----------
        x_select: str
        y_selects: tuple[str] or list[str]
            path select

        Returns
        -------
        x_data: tuple(str, list)
            x_key: str
                代表 x 的名称
            x: list
                代表数据
        y_datas: list[list[tuple(str, list)]]
            每一个元素都是一个列表，列表中的每一个元素是一个元组，元组包含两个元素:
                第一个元素，字符串类型，代表数据名称(path_key形式)
                第二个元素，列表，代表数据
        """
        x_pattern = re.compile(x_select)
        y_patterns = [re.compile(y_select) for y_select in y_selects]

        x = None
        x_key = None
        ys = [[] for _ in y_patterns]
        for key, value in self.items():
            if x_pattern.match(key):
                assert x is None, "x has been set, duplicated x, %s [stored] vs %s [store]" % (x_key, key)
                x_key = key
                x = value
            else:
                for i, y_pattern in enumerate(y_patterns):
                    if y_pattern.match(key):
                        ys[i].append((key, value))

        return (x_key, x), ys

    def visual_select(self, x_select='iteration', y_select='accuracy|prf_avg_.*'):
        """
        按 x，y 格式选取数据，x_select 和 y_select 都是正则匹配式，采用 re.match 方式匹配

        Parameters
        ----------
        x_select: str
        y_select: str
            path select

        Returns
        -------
        x_data: tuple(str, list)
            x_key: str
                代表 x 的名称
            x: list
                代表数据
        y_datas: list[tuple(str, list)]
            每一个元素都是一个元组，元组包含两个元素:
                第一个元素，字符串类型，代表数据名称(path_key形式)
                第二个元素，列表，代表数据
        """
        x_pattern = re.compile(x_select)
        y_pattern = re.compile(y_select)

        x = None
        x_key = None
        ys = []
        for key, value in self.items():
            if x_pattern.match(key):
                assert x is None, "x has been set, duplicated x, %s [stored] vs %s [store]" % (x_key, key)
                x_key = key
                x = value
            else:
                if y_pattern.match(key):
                    ys.append((key, value))

        return (x_key, x), ys


def universe(result):
    """

    Parameters
    ----------
    result: ResultAnalyser

    """
    (x_key, x), ys = result.visual_select(y_select='accuracy|prf_avg_.*')
    plt.figure()
    plt.title('universe')
    for y_key, y in ys:
        plt.plot(x, y, label=y_key.split('_')[-1])

    plt.xlabel(x_key)
    plt.legend(labels=plt.get_figlabels())
    plt.show()


def bd(num):
    for i in range(int(math.ceil(math.sqrt(num))), 1, -1):
        if num % i == 0:
            return i, num // i
    return 1, num


def prf(result, class_num, figure_max_class=8):
    """

    Parameters
    ----------
    result: ResultAnalyser

    """
    import math
    (x_key, x), yss = result.visual_selects(y_selects=[r'prf_%s_.*' % class_id for class_id in range(class_num)])
    figure_max_class = class_num if not figure_max_class else figure_max_class
    row, col = bd(figure_max_class)
    figures = [plt.figure(i) for i in range(int(math.floor(class_num / figure_max_class)))]
    subplots = [figures[i].subplots(row, col, sharex='all', squeeze=False).flat for i in range(len(figures))]
    for ys in yss:
        for y_key, y in ys:
            class_id = int(ys[0][0].split('_')[1])
            sp = subplots[class_id // figure_max_class][class_id % figure_max_class]
            sp.set_title('class_' + str(class_id))
            sp.plot(x, y, label=y_key.split('_')[-1])
            plt.xlabel(x_key)

    for sps in subplots:
        for sp in sps:
            handles, labels = sp.get_legend_handles_labels()
            sp.legend(handles, labels)

    for figure in figures:
        figure.tight_layout()
    plt.show()


def prf_group(result, detailed=True):
    (x_key, x), yss = result.visual_selects(
        y_selects=['accuracy|prf_avg_.*', 'prf_\d+_precision', 'prf_\d+_recall', 'prf_\d+_f1'])



def precision_group(result):
    """

    Parameters
    ----------
    result: ResultAnalyser

    """
    (x_key, x), ys = result.visual_select(y_select='prf_\d+_precision')
    plt.figure()
    plt.title('precision')
    for y_key, y in ys:
        label = y_key.split('_')[1]
        plt.plot(x, y, label=label)

    plt.xlabel(x_key)
    plt.ylabel(r'%')

    axes = plt.gca()
    handles, labels = axes.get_legend_handles_labels()
    lines, labels = zip(*sorted(zip(handles, labels), key=lambda x: int(x[1])))
    plt.legend(lines, labels, ncol=math.ceil(len(labels) / 8))
    plt.show()


def recall_group(result):
    pass


def f1_group(result):
    pass


def precision():
    pass


if __name__ == '__main__':
    import json

    result = ResultAnalyser()
    with open("result1.json") as f:
        for line in f:
            result.add_record(json.loads(line))
    precision_group(result)

    # print(plt)
