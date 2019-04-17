# coding: utf-8
# create by tongshiwei on 2018/7/18


from dev.framework.KG.dataset.universe import mapper_transform
from dev.framework.KG.io_lib import UniteMapper


def unite_transform(sources, targets, unite_map_filename):
    unite_map, unite_size = UniteMapper.load(unite_map_filename)
    unite_mapper = UniteMapper(unite_map, unite_size)
    mapper_transform(sources, targets, unite_mapper)

