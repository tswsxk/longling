# coding:utf-8
# created by tongshiwei on 2018/7/26
"""
一个简单的装置文件
目前只对二维数据使用
两种格式：
Row:
    [feature0[0], feature1[0],... featurem[0]]
    ...
    [feature0[n], feature1[n],... featuren[n]]
Col:
    [
        feature0[0..n],
        feature1[0..n],
        ...
        featurem[0..n]
    ]
Row格式写入的时候比较好组织，可以在单一文件内实现生成器读入
Col格式处理起来容易，无法实现在单一文件内实现生成器读入
"""


import json

from tqdm import tqdm

from longling.lib.stream import wf_open
from longling.lib.candylib import as_list


def load_row(filenames, line_parse=lambda x: x.split()):
    """
    装载以Row格式存储的文件
    Parameters
    ----------
    filenames: str or list[str]
        Row格式文件
    line_parse: function
        如何解析文件
    Returns
    -------

    """
    for filename in as_list(filenames):
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield line_parse(line)


def load_col(filenames, line_parse=lambda x: json.loads(x)):
    """
    装载以Col格式存储的文件
    每一行或每一个文件的第一行存储一个特征
    Parameters
    ----------
    filenames: str or list[str]
        Col格式文件
    line_parse: function
        如何解析文件
    Returns
    -------

    """
    filenames = as_list(filenames)
    if len(filenames) == 1:
        filename = filenames[0]
        features = []
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                feature = line_parse(line)
                features.append(feature)
    else:
        features = []
        for filename in filenames:
            with open(filename) as f:
                features.append(json.loads(f.readline()))
    return features


def row_file2col_files(row_filenames, col_filenames, feature_num, line_parse=lambda x: x.split()):
    """
    将存储有row格式数据的文件，转换为col格式的文件
    Parameters
    ----------
    row_filenames: str or list[str]
    col_filenames: str or list[str]
    feature_num: int
    line_parse: function

    Returns
    -------

    """
    feature_datas = [[] for _ in range(feature_num)]
    for features in tqdm(load_row(row_filenames, line_parse)):
        assert len(features) == len(feature_datas)
        for i, feature in enumerate(features):
            feature_datas[i].append(feature)
    col_filenames = as_list(col_filenames)
    if len(col_filenames) == 1:
        # generate the data file in col type using this way will disable the way to use generator
        with wf_open(col_filenames[0]) as wf:
            for feature_data in feature_datas:
                print(json.dumps(feature_data, ensure_ascii=False), file=wf)
    elif len(col_filenames) == len(feature_datas):
        for col_filename, feature_data in zip(col_filenames, feature_datas):
            with wf_open(col_filename) as wf:
                print(json.dumps(feature_data, ensure_ascii=False), file=wf)
    else:
        raise AssertionError("len(col_filenames) == 1 or len(col_filenames) == len(feature_datas)")


def row2col(row_datas, feature_num=None):
    """
    转置
    将row格式的数据转换为col格式的数据
    Parameters
    ----------
    row_datas: iterable
        row 格式的数据,
    feature_num: int
        特征数
    Returns
    -------
    col 格式数据
    """
    if feature_num is None:
        feature_datas = [[data] for data in next(row_datas)]
    else:
        feature_datas = [[] for _ in range(feature_num)]
    for row_data in tqdm(row_datas, "row2col"):
        if not row_data:
            continue
        features = row_data
        assert len(features) == len(feature_datas)
        for i, feature in enumerate(features):
            feature_datas[i].append(feature)
    return feature_datas
