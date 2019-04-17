# coding: utf-8
# create by tongshiwei on 2018/7/18
from dev.framework.KG.dataset.universe import mapper_transform
from dev.framework.KG.io_lib import ERMapper


def er_transform(sources, targets, entities_map_filename, relations_map_filename):
    entities_map, relations_map, entities_size, relations_size = ERMapper.load(entities_map_filename,
                                                                               relations_map_filename)
    er_mapper = ERMapper(entities_map, relations_map, entities_size, relations_size)
    mapper_transform(sources, targets, er_mapper)
