# coding:utf-8
# created by tongshiwei on 2018/7/25

# coding:utf-8
# created by tongshiwei on 2018/7/22
"""
适用于RNN的bucket方法
需要预先将数据按bucket排序好再调用此方法，进行bucket和padding
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

返回的bucket内的数据都用Col格式
即
buckets = [
    bucket0[
        feature0[0..n],
        ...
        featurem[0..n]
    ],
    ...
    bucketk[
        feature0[0..n],
        ...
        featurem[0..n]
    ]
]
Bucket 的一般流程
原始文件排序 -> bucket -> padding
"""
import json

from tqdm import tqdm

from longling.lib.stream import wf_open
from longling.lib.candylib import as_list


def load_row(filename, line_parse=lambda x: x.split()):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield line_parse(line)


def load_col(filenames):
    filenames = as_list(filenames)
    if len(filenames) == 1:
        filename = filenames[0]
        features = []
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                feature = json.loads(line)
                features.append(feature)
    else:
        features = []
        for filename in filenames:
            with open(filename) as f:
                features.append(json.loads(f.readline()))
    return features


def row_file2col_files(row_filename, col_filenames, feature_num, line_parse=lambda x: x.split()):
    """
    将存储有row格式数据的文件，转换为col格式的文件
    Parameters
    ----------
    row_filename: str
    col_filenames: str or list[str]
    feature_num: int
    line_parse: function

    Returns
    -------

    """
    feature_datas = [[] for _ in range(feature_num)]
    for features in tqdm(load_row(row_filename, line_parse)):
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


class Bucket(object):
    """
    篮子类
    将数据转变为 bucket 格式
    方便 RNN 操作
    """

    def __init__(self, source_datas, bucket_size,
                 bucket_padding='</s>', ignore_padding_index=-1,
                 is_source_datas_generator=False,
                 ):
        """

        Parameters
        ----------
        source_datas: list or generator
        bucket_size: int
        bucket_padding: str or list np.array
        ignore_padding_index: int
        is_source_datas_generator: bool
        """
        self.bucket_padding = bucket_padding
        self.ignore_padding_index = ignore_padding_index
        self.bucket_size = bucket_size
        self.is_sd_gen = is_source_datas_generator
        self.buckets = self.bucket(source_datas)

    def bucket(self, source_datas):
        raise NotImplementedError

    @staticmethod
    def gen_fetch(generator, num):
        datas = []
        for i in range(num):
            try:
                datas.append(next(generator))
            except StopIteration:
                break
        return datas

    def __iter__(self):
        return self.buckets

    def __next__(self):
        return next(self.buckets)


class ColBucket(Bucket):
    """
    针对 Col 数据格式的篮子类
    生成器
    """

    def bucket(self, source_datas, total_num=None):
        """

        Parameters
        ----------
        source_datas: list[list] or generator
        total_num: int or None

        Returns
        -------

        """
        bucket = []
        if not self.is_sd_gen:
            total_num = len(source_datas[0]) if total_num is None else total_num
            for i in range(self.bucket_size, total_num, self.bucket_size):
                bucket.append([feature[i: i + self.bucket_size] for feature in source_datas])
                yield bucket
            raise StopIteration
        else:
            assert total_num, "when data sources contains generator, total_num is required"
            for i in range(self.bucket_size, total_num, self.bucket_size):
                bucket.append([self.gen_fetch(feature, self.bucket_size) for feature in source_datas])
                yield bucket
            raise StopIteration


class RowBucket(Bucket):
    def bucket(self, source_datas):
        """
        针对Row格式的篮子类
        生成器
        Parameters
        ----------
        source_datas: list[list] or generator

        Returns
        -------

        """
        if self.is_sd_gen:
            while True:
                source_data = self.gen_fetch(source_datas, self.bucket_size)
                if not source_data:
                    raise StopIteration
                features_datas = [[] for _ in range(len(source_data[0]))]
                for features in source_data:
                    for i, feature in enumerate(features):
                        features_datas[i].append(feature)
                yield features_datas
        else:
            total_num = len(source_datas)
            bucket_features = [[] for _ in source_datas[0]]
            for i in range(self.bucket_size, total_num, self.bucket_size):
                for feature_datas in source_datas[i: i + self.bucket_size]:
                    for j, feature in enumerate(feature_datas):
                        bucket_features[j].append(feature)
                yield bucket_features
            raise StopIteration
