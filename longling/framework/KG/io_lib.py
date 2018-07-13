# coding:utf-8
# created by tongshiwei on 2018/7/9

import json
from collections import defaultdict

from tqdm import tqdm

from longling.lib.stream import wf_open, wf_close


def load_plain(rdf_txt):
    """
    rdf txt 文件的每一行存储一个三元组，空格分隔
    Parameters
    ----------
    rdf_txt

    Returns
    -------

    """
    rdf_triples = []
    with open(rdf_txt) as f:
        for line in tqdm(f, rdf_txt):
            line = line.strip()
            if line:
                rdf_triple = line.split()
                assert len(rdf_triple) == 3
                rdf_triples.append(rdf_triple)
    return rdf_triples


def load_plains(rdf_txt_files):
    rdf_triples = []
    for rdf_txt in rdf_txt_files:
        rdf_triples.extend(load_plain(rdf_txt))
    return rdf_triples


def rdf2sro(rdf_triples):
    sro = {}
    entities = set()
    relations = set()
    for rdf_triple in tqdm(rdf_triples):
        entities.add(rdf_triple[0])
        entities.add(rdf_triple[2])
        relations.add(rdf_triple[1])

        if rdf_triple[0] not in sro:
            sro[rdf_triple[0]] = {}
        if rdf_triple[1] not in sro[rdf_triple[0]]:
            sro[rdf_triple[0]][rdf_triple[1]] = set()
        sro[rdf_triple[0]][rdf_triple[1]].add(rdf_triple[2])

    return sro, entities, relations


def rdf2rso(rdf_triples):
    rso = {}
    entities = set()
    relations = set()
    for rdf_triple in tqdm(rdf_triples):
        entities.add(rdf_triple[0])
        entities.add(rdf_triple[2])
        relations.add(rdf_triple[1])

        if rdf_triple[1] not in rso:
            rso[rdf_triple[1]] = {}
        if rdf_triple[2] not in rso[rdf_triple[1]]:
            rso[rdf_triple[1]][rdf_triple[0]] = set()
        rso[rdf_triple[1]][rdf_triple[0]].add(rdf_triple[2])

    return rso, entities, relations


def rdf2ors(rdf_triples):
    rso = {}
    entities = set()
    relations = set()
    for rdf_triple in tqdm(rdf_triples):
        entities.add(rdf_triple[0])
        entities.add(rdf_triple[2])
        relations.add(rdf_triple[1])

        if rdf_triple[2] not in rso:
            rso[rdf_triple[2]] = {}
        if rdf_triple[1] not in rso[rdf_triple[2]]:
            rso[rdf_triple[2]][rdf_triple[1]] = set()
        rso[rdf_triple[2]][rdf_triple[1]].add(rdf_triple[0])

    return rso, entities, relations


def plain2sro(loc_plain, loc_sro):
    rdf_triples = load_plain(loc_plain)
    sro, _, _ = rdf2sro(rdf_triples)

    with wf_open(loc_sro) as wf:
        for k, v in tqdm(sro.items()):
            print(json.dumps((k, v), ensure_ascii=False), file=wf)


class Mapper(object):
    def transform(self, source, target):
        raise NotImplementedError()

    @staticmethod
    def get_dict(filename):
        raise NotImplementedError

    def save(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def load(*args, **kwargs):
        raise NotImplementedError


class ERMapper(Mapper):
    def __init__(self, entities_map=None, relations_map=None, entities_size=None, relations_size=None,
                 base_filename=None):
        if entities_map and relations_map and entities_size and relations_size:
            self.entities_map = entities_map
            self.relations_map = relations_map
            self.entities_size = entities_size
            self.relations_size = relations_size
        else:
            assert base_filename
            entities_map, relations_map, entities_size, relations_size = self.get_dict(base_filename)
            self.entities_map = entities_map
            self.relations_map = relations_map
            self.entities_size = entities_size
            self.relations_size = relations_size

    @staticmethod
    def get_dict(filename):
        entities_map, e_idx = defaultdict(int), 1
        relations_map, r_idx = defaultdict(int), 1
        with open(filename) as f:
            for line in tqdm(f, desc="build dict from file[%s]" % filename):
                if not line.strip():
                    continue
                rdf_triple = line.split()
                assert len(rdf_triple) == 3
                sub, rel, obj = rdf_triple
                if sub not in entities_map:
                    entities_map[sub] = e_idx
                    e_idx += 1
                if obj not in entities_map:
                    entities_map[obj] = e_idx
                    e_idx += 1
                if rel not in relations_map:
                    relations_map[rel] = r_idx
                    r_idx += 1
        return entities_map, relations_map, e_idx, r_idx

    def transform(self, source, target):
        wf = wf_open(target)
        with open(source) as f:
            for line in tqdm(f, "%s transform %s to %s" % (self.__class__.__name__, source, target)):
                line = line.strip()
                if line:
                    rdf_triple = line.split()
                    assert len(rdf_triple) == 3
                    print("%s\t%s\t%s" % (self.entities_map[rdf_triple[0]], self.relations_map[rdf_triple[1]],
                                          self.entities_map[rdf_triple[2]]), file=wf)
        wf_close(wf)

    def save(self, entities_map_filename, relations_map_filename):
        wf = wf_open(entities_map_filename)
        json.dump(self.entities_map, wf, ensure_ascii=False)
        wf_close(wf)
        wf = wf_open(relations_map_filename)
        json.dump(self.relations_map, wf, ensure_ascii=False)
        wf_close(wf)

    @staticmethod
    def load(entities_map_filename, relations_map_filename):
        with open(entities_map_filename) as f:
            entities_map = defaultdict(int)
            entities_map.update(json.load(f))
            entities_size = len(entities_map) + 1
        with open(relations_map_filename) as f:
            relations_map = defaultdict(int)
            relations_map.update(json.load(f))
            relations_size = len(relations_map) + 1

        return entities_map, relations_map, entities_size, relations_size

