# coding: utf-8
# create by tongshiwei on 2019/5/25

"""
适合大规模数据的特征编码器，
用以进行特征编码，即 new_feature = encoder(feature)。
是pandas，sklearn等通用特征编码的补充类，
主要应用场景是数据无法一次性装载到内存中。

所有 Encoder 和 其子类 都包含五个主要方法

* add: 添加元素
* fit: 拟合
* transform: 元素转换
* fit_transform: 拟合并转换
* __call__: 元素转换，默认为 transform函数

思路和 sklearn.preprocess 方法类似，
不同之处在于，fit 的添加元素功能分离到 add 中。
在调用fit函数前需要使用 add 方法添加元素。
"""

import json
import pickle

from longling.DM.structure.MapperMeta import MapperMeta


class Encoder(MapperMeta):

    def __init__(self, mapper_meta, fit_required=False, *args, **kwargs):
        super(Encoder, self).__init__(mapper_meta)
        self.fit_required = fit_required

    def add(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def transform(self, *args, **kwargs):
        raise NotImplementedError

    def fit_transform(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    @staticmethod
    def as_list(obj):
        if isinstance(obj, list):
            return obj
        elif isinstance(obj, tuple):
            return list(obj)
        else:
            return [obj]


class LambdaEncoder(Encoder):
    """
    简单的基于lambda函数的转换器，常用的转换函数可以是：

    * 输入转整型 lambda x: int(x)
    * 输入转浮点 lambda x: float(x)
    * 值转换 lambda x: -1 if x == "" else x
    """

    def __init__(self, mapper_meta=None, lambda_func=lambda x: x):
        mapper_meta = mapper_meta if mapper_meta else [
            lambda_func, str(pickle.dumps(lambda_func))
        ]
        super(LambdaEncoder, self).__init__(mapper_meta, False)

    def dump2one(self, fp, indent=2, sort_keys=False, **kwargs):
        json.dump(
            [
                self.meta_instance,
                self.meta_instance
            ],
            fp=fp,
            indent=indent,
            sort_keys=sort_keys,
            **kwargs
        )

    @staticmethod
    def load4one(fp, **kwargs):
        mapper_meta = super().load4one(fp)
        mapper_meta[0] = bytes(mapper_meta[0])
        return mapper_meta

    @staticmethod
    def retrieve4one(raw_mapper_meta):
        mapper = pickle.loads(bytes(raw_mapper_meta[0]))
        return [mapper, raw_mapper_meta[1]]

    def transform(self, values):
        for value in self.as_list(values):
            yield self.mapper(value)

    def add(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass

    def fit_transform(self, values):
        return self.transform(values)


class OrdinalEncoder(Encoder):
    """
    将特征转换为定性特征值，每一种特征都当成一类
    """

    def __init__(self, mapper_meta=None):
        mapper_meta = mapper_meta if mapper_meta else [
            {},
            0,
        ]
        super(OrdinalEncoder, self).__init__(mapper_meta, False)

    @property
    def meta(self):
        return self.mapper_meta[1]

    @meta.setter
    def meta(self, value):
        self.mapper_meta[1] = value

    def add(self, values):
        for value in self.as_list(values):
            if value not in self.mapper:
                self.mapper[value] = self.meta
                self.meta += 1

    def fit(self):
        pass

    def transform(self, values):
        for value in self.as_list(values):
            yield self.mapper[value]

    def fit_transform(self, values):
        values = self.as_list(values)
        for value in values:
            self.add(value)
        self.fit()
        for value in values:
            yield self.transform(value)


class EncoderGroup(Encoder):
    """
    编码器组，适合用来对不同列的数据进行不同方式的转换
    """

    def __init__(self,
                 mapper_meta=None,
                 default_encoder_type=OrdinalEncoder.__name__
                 ):
        super(EncoderGroup, self).__init__(mapper_meta)
        self.default_encoder_type = default_encoder_type

    @staticmethod
    def retrieve4one(raw_mapper_meta):
        mapper_meta = [
            {},
            {},
        ]
        mapper, meta = raw_mapper_meta
        for feature_index in mapper:
            feature_type = meta.get(feature_index, OrdinalEncoder.__name__)
            mapper_meta[0][feature_index] = eval(feature_type).retrieve4one(
                mapper[feature_index]
            )
            mapper_meta[1][feature_index] = feature_type
        return mapper_meta

    @property
    def mapper_instance(self):
        _mapper_instance = {}
        for feature_name in self.mapper:
            _mapper_instance[feature_name] = self.mapper[
                feature_name].mapper_meta
        return _mapper_instance

    def set_feature_type(self, feature_index, feature_type):
        assert feature_index not in self, \
            "The feature already exists, cannot change its feature type. " \
            "Current feature type is %s" % self.mapper[feature_index].__name__

        assert isinstance(feature_type, Encoder), \
            "feature_type should be Encoder, now is %s" % type(feature_type)
        self.mapper[feature_index] = feature_type
        self.meta[feature_index] = type(feature_type).__name__

    def add(self, feature_index, values):
        if feature_index not in self:
            self.mapper[feature_index] = eval(
                self.meta.get(feature_index, self.default_encoder_type)
            )()
        self.mapper[feature_index].add(values)

    def fit(self):
        for feature_name in self.mapper:
            self.mapper[feature_name].fit()

    def transform(self, feature_index, values):
        return self.mapper[feature_index].transform(values)

    def fit_transform(self, values):
        if isinstance(values, dict):
            return {
                feature_index: self.mapper[feature_index].fit_transform(
                    values[feature_index]
                ) for feature_index in values
            }
        else:
            return [
                self.mapper[feature_index].fit_transform(
                    _values
                ) for feature_index, _values in enumerate(values)
            ]

    def __call__(self, feature_index, value):
        return self.transform(feature_index, value)
