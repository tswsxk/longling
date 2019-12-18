# coding: utf-8
# create by tongshiwei on 2018/8/16
"""
此模块用来针对result.json中的数据进行数据分析
开发测试中，非稳定版本
"""
import math
import re
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt

__all__ = ["ResultAnalyser"]

warnings.warn("do not use this package, unstable")


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
        添加字典数据并解析，如果字典的value也是字典的话，会深入解析，
        键名会进行拼接操作，如: {'key1': {'key2': value}}
        会返回 (key1_key2, value)

        Parameters
        ----------
        dict_data: dict
            字典格式数据
        prefix: str
        """
        for key, value in dict_data.items():
            if isinstance(value, dict):
                self.add_record(value, prefix=prefix + '_' + str(
                    key) if prefix else str(key))
            else:
                self.records[
                    prefix + '_' + str(key) if prefix else str(key)].append(
                    value)

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

    def visual_selects(self, x_select='iteration',
                       y_selects=('accuracy|prf_avg_.*', 'prf_\d+_f1')):
        """
        按 x，y 格式选取数据，x_select 和 y_selects 都是正则匹配式，
        采用 re.match 方式匹配

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
            每一个元素都是一个列表，列表中的每一个元素是一个元组，
            元组包含两个元素:
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
                assert x is None, \
                    "x has been set, duplicated x, " \
                    "%s [stored] vs %s [store]" % (
                        x_key, key
                    )
                x_key = key
                x = value
            else:
                for i, y_pattern in enumerate(y_patterns):
                    if y_pattern.match(key):
                        ys[i].append((key, value))

        return (x_key, x), ys

    def visual_select(self, x_select='iteration',
                      y_select='accuracy|prf_avg_.*'):
        """
        按 x，y 格式选取数据，x_select 和 y_select 都是正则匹配式，
        采用 re.match 方式匹配

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
                assert x is None, "x has been set, duplicated x, " \
                                  "%s [stored] vs %s [store]" % (
                                      x_key, key
                                  )
                x_key = key
                x = value
            else:
                if y_pattern.match(key):
                    ys.append((key, value))

        return (x_key, x), ys


def method_compare(method_args, x_select='iteration',
                   y_select='accuracy|prf_avg_.*|.*loss.*', last_only=True):
    """
    @ Dev

    对比不同方法在不同指标下的表现，两种呈现方式，
    一种以表的形式，一种以图的形式
    在图形式下，如果有多个指标，将会绘制多个子图
    # todo 统一相同label的线的颜色
    # todo 处理x_select出的x长度不一致的case

    Parameters
    ----------
    method_args: list[tuple(method_name, filename)]
    x_select: str
    y_select: str
    last_only: bool
        只返回每个指标的最终值
        是的话，以pandas.DataFrame的形式返回一个表，
        # todo 如果没有 pandas 包, 将返回一个字典，
        此时 x_select 将被无视
        否则，进行可视化

    Returns
    -------

    """
    import json
    key_results = {}

    x_key = None
    x = None
    for model_name, filename in method_args:
        result = ResultAnalyser()
        with open(filename) as f:
            for line in f:
                result.add_record(json.loads(line))
        (x_key, x), ys = result.visual_select(x_select, y_select)
        for y_key, y in ys:
            y_key = y_key.replace('prf_avg_', '')
            if y_key not in key_results:
                key_results[y_key] = {}
            if last_only:
                key_results[y_key][model_name] = y[-1]
            else:
                key_results[y_key][model_name] = y
    if last_only:
        import pandas
        return pandas.DataFrame.from_dict(key_results)

    figure = plt.figure()
    row, col = bd(len(key_results))
    axes = figure.subplots(row, col, sharex='all').flat
    for i, (key, class_values) in enumerate(key_results.items()):
        for class_name, class_value in class_values.items():
            axes[i].plot(x, class_value, label=class_name)
        axes[i].set_title(key)
        axes[i].set_xlabel(x_key)
        handles, labels = axes[i].get_legend_handles_labels()
        lines, labels = zip(*sorted(zip(handles, labels), key=lambda x: x[1]))
        axes[i].legend(lines, labels)
    figure.tight_layout()
    plt.show()


def universe(result, select='accuracy|prf_avg_.*|.*loss.*'):
    """
    平均指标可视化

    Parameters
    ----------
    result: ResultAnalyser

    select: str

    """
    (x_key, x), ys = result.visual_select(y_select=select)
    plt.figure()
    plt.title('universe')
    for y_key, y in ys:
        plt.plot(x, y, label=y_key.split('_')[-1])
        # todo 添加最终结果的线annotation
        # plt.plot([0, x[-1]], [y[-1], y[-1]], '--')
        # todo 添加收敛位置的线annotation

    plt.xlabel(x_key)

    plt.legend()

    plt.show()


def bd(num):
    """
    合数最大分解

    Parameters
    ----------
    num: int

    Returns
    -------

    """
    for i in range(int(math.ceil(math.sqrt(num))), 1, -1):
        if num % i == 0:
            return i, num // i
    return 1, num


def prf(result, class_num, figure_max_class=8, class_key_map=None):
    """
    @分类问题
    分别显示每个类的prf图
    每个子图包含三条线：precision, recall, f1

    Parameters
    ----------
    result: ResultAnalyser
        结果记录
    class_num: int
        类别总数
    figure_max_class: int
        每一幅图可包含的最大子图数
    class_key_map: dict{int: *}
        类名映射字典
    """
    import math

    def key_name(key):
        if class_key_map is not None:
            return class_key_map[key]
        else:
            return str(key)

    (x_key, x), yss = result.visual_selects(
        y_selects=[r'prf_%s_.*' % class_id for class_id in range(class_num)])
    figure_max_class = class_num if not figure_max_class else figure_max_class
    row, col = bd(figure_max_class)
    figures = [plt.figure(i) for i in
               range(int(math.floor(class_num / figure_max_class)))]
    subplots = [figures[i].subplots(row, col, sharex='all', squeeze=False).flat
                for i in range(len(figures))]
    for ys in yss:
        for y_key, y in ys:
            class_id = int(ys[0][0].split('_')[1])
            sp = subplots[class_id // figure_max_class][
                class_id % figure_max_class]
            sp.set_title('class_' + key_name(class_id))
            sp.plot(x, y, label=y_key.split('_')[-1])
            plt.xlabel(x_key)

    for sps in subplots:
        for sp in sps:
            handles, labels = sp.get_legend_handles_labels()
            sp.legend(handles, labels)

    for figure in figures:
        figure.tight_layout()
    plt.show()


def evaluation_group(result, select):
    """

    Parameters
    ----------
    result: ResultAnalyser
    select: str

    Returns
    -------

    """
    # todo 统一不同子图间相同label的颜色
    (x_key, x), ys = result.visual_select(y_select=select)
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


def precision_group(result):
    """

    Parameters
    ----------
    result: ResultAnalyser

    """
    evaluation_group(result, "prf_\d+_precision")


def recall_group(result):
    """

    Parameters
    ----------
    result: ResultAnalyser

    """
    evaluation_group(result, r"prf_\d+_precision")


def f1_group(result):
    """

    Parameters
    ----------
    result: ResultAnalyser

    """
    evaluation_group(result, r"prf_\d+_f1")


def pandas_api(result):
    # todo use pandas axes to form the graph
    """try to plot the a1 and a2 on the same figure,
    but now will have three figure, and target figure empty"""
    import pandas
    import numpy as np
    (x_key, x), yss = result.visual_select()
    column, datas = zip(*yss)
    f = plt.figure()
    a1 = pandas.DataFrame.from_records(
        np.asarray(list(datas)).T, columns=column
    )
    b = a1.plot()
    f.axes.append(b)
    a2 = pandas.DataFrame.from_records(
        np.asarray(list(datas)).T, columns=column
    )
    b = a2.plot()
    f.axes.append(b)
    f.canvas.draw()
    plt.show()


if __name__ == '__main__':
    print(method_compare(
        method_args=[('m1', 'result1.json'), ('m2', 'result2.json')]
    ))
    method_compare(
        method_args=[('m1', 'result1.json'), ('m2', 'result2.json')],
        last_only=False
    )
