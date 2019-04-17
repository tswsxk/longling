# coding: utf-8
# create by tongshiwei on 2018/7/18
from tqdm import tqdm

from longling.lib.candylib import as_list
from dev.framework.KG.io_lib import UniteMapper, ERMapper, SROMapper


def build_dict(base_filename, mapper_type, mapper_args):
    assert mapper_type in (UniteMapper, ERMapper, SROMapper)
    mapper_type(base_filename=base_filename)
    mapper_type(*mapper_args)


def mapper_transform(sources, targets, mapper):
    for source, target in zip(as_list(sources), as_list(targets)):
        mapper.transform(source, target)


def get_unite(rdf_triples):
    entities, relations = get_er(rdf_triples)
    return entities.union(relations)


def get_sro(rdf_triples):
    subjects = set()
    relations = set()
    objects = set()
    for rdf_triple in tqdm(rdf_triples):
        subjects.add(rdf_triple[0])
        relations.add(rdf_triple[1])
        objects.add(rdf_triple[2])
    return subjects, relations, objects


def get_er(rdf_triples):
    subjects, relations, objects = get_sro(rdf_triples)
    return subjects.union(objects), relations
