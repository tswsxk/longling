# coding:utf-8
# created by tongshiwei on 2018/7/25
"""
适用于RNN的bucket方法
需要预先将数据按bucket排序好再调用此方法，进行bucket和padding

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


class IterBucket(object):
    """
    桶类
    将数据转变为 bucket 格式
    方便 RNN 操作
    """

    def __init__(self, source_datas, bucket_size,
                 bucket_padding='</s>', ignore_padding_index=-1,
                 is_source_datas_iterator=False,
                 ):
        """

        Parameters
        ----------
        source_datas: list or generator
        bucket_size: int
        bucket_padding: str or list np.array
        ignore_padding_index: int
        is_source_datas_iterator: bool
        """
        self.bucket_padding = bucket_padding
        self.ignore_padding_index = ignore_padding_index
        self.bucket_size = bucket_size
        self.is_sd_iter = is_source_datas_iterator
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


class ColIterBucket(IterBucket):
    """
    针对 Col 数据格式的桶类
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
        if not self.is_sd_iter:
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


class RowIterBucket(IterBucket):
    def bucket(self, source_datas):
        """
        针对Row格式的桶类
        生成器
        Parameters
        ----------
        source_datas: list[list] or generator

        Returns
        -------

        """
        if self.is_sd_iter:
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
