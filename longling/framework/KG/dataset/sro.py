# coding: utf-8
# create by tongshiwei on 2018/7/18
from tqdm import tqdm

from longling.framework.KG.dataset.universe import mapper_transform
from longling.framework.KG.io_lib import SROMapper


def sro_transform(sources, targets, subjects_map_filename, relations_map_filename, objects_map_filename):
    subjects_map, relations_map, objects_map, entities_size, relations_size, objects_size = SROMapper.load(
        subjects_map_filename, relations_map_filename, objects_map_filename)
    sro_mapper = SROMapper(subjects_map, relations_map, objects_map, entities_size, relations_size, objects_size)
    mapper_transform(sources, targets, sro_mapper)
