# coding: utf-8
# create by tongshiwei on 2019/5/25

__all__ = ["MapperMeta"]

import json
import pickle


class MapperMeta(object):
    """
    一种映射器包装类，方便将映射器进行持久化

    包括两个主元素：
    * mapper: 映射器，映射器可以是字典、函数或类
    * meta: 映射器元数据，可以是生成映射器必要的参数，也可是一些描述

    主要包括六个属性：
    * mapper_meta: list， 存储映射器-元数据，
        一般包括两个元素，第一个为映射器，第二个为元数据
    * mapper: 映射器
    * meta: 元数据
    * instance: mapper_meta 的持久化对象，
        可在这里将mapper_meta中不可持久化，或不需要持久化的内容去掉
    * mapper_instance: mapper 的持久化对象
    * meta_instance: meta 的持久化对象

    提供两类基本的持久化方法:
    * One: 将 mapper_meta 持久化到一个文件中，包括三个相关方法：
        * dump2one: 将 instance 属性 dump 到指定文件中，默认是 json 格式
        * load4one: 从指定文件中 load 出 raw_mapper_meta，默认是 json 格式
        * retrieve4one: 对使用 load4one 得到的 raw_mapper_meta 进行进一步处理，
            默认不处理，返回 raw_mapper_meta
    * Multi: 将 mapper_meta 持久化到多个文件中，一个存储 mapper，一个存储 meta，
        包括五个相关方法：
        * dump: 调用 mapper_dump 和 meta_dump （见下）分别将
            mapper_instance 和 meta_instance dump 到不同的文件中
        * mapper_dump: 将 mapper_instance dump 到指定文件中，默认是 pickle 格式
        * meta_dump: 将 meta_instance dump 到指定文件中，默认是 json 格式
        * mapper_load: 从指定文件中 load 出 mapper，默认是 pickle 格式
        * meta_load: 从指定文件中 load 出 meta，默认是 json 格式
    """

    def __init__(self, mapper_meta=None):
        self.mapper_meta = mapper_meta if mapper_meta else [None, None]

    def __contains__(self, item):
        return item in self.mapper

    def __iter__(self):
        return self.mapper.__iter__()

    # ######################### Property ###################################
    @property
    def mapper(self):
        return self.mapper_meta[0]

    @property
    def meta(self):
        return self.mapper_meta[1]

    @property
    def instance(self):
        return [self.mapper_instance, self.meta_instance]

    @property
    def mapper_instance(self):
        return self.mapper

    @property
    def meta_instance(self):
        return self.meta

    # ######################### The persistence methods #####################
    # ######################### Single #########################
    def dump2one(self, fp, indent=2, sort_keys=False, **kwargs):
        json.dump(
            self.instance,
            fp,
            indent=indent,
            sort_keys=sort_keys,
            **kwargs
        )

    @staticmethod
    def retrieve4one(raw_mapper_meta):
        return raw_mapper_meta

    @staticmethod
    def load4one(fp, **kwargs):
        return json.load(fp, **kwargs)

    # ######################### Multi #############################
    def dump(self, mapper_fp, meta_fp, mapper_kwargs, meta_kwargs):
        self.mapper_dump(mapper_fp, **mapper_kwargs)
        self.meta_dump(meta_fp, **meta_kwargs)

    def mapper_dump(self, fp, **kwargs):
        pickle.dump(self.mapper_instance, fp, **kwargs)

    @staticmethod
    def mapper_load(fp, **kwargs):
        return pickle.load(fp, **kwargs)

    def meta_dump(self, fp, **kwargs):
        json.dump(self.meta_instance, fp, **kwargs)

    @staticmethod
    def meta_load(fp, **kwargs):
        return json.load(fp, **kwargs)
