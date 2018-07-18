# coding: utf-8
# create by tongshiwei on 2018/7/18

import json
from collections import defaultdict

from tqdm import tqdm

from longling.lib.stream import wf_open, wf_close


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


class SROMapper(Mapper):
    def __init__(self, subjects_map=None, relations_map=None, objects_map=None,
                 subjects_size=None, relations_size=None, objects_size=None,
                 base_filename=None):
        if subjects_map and relations_map and subjects_size and relations_size:
            self.subjects_map = subjects_map
            self.relations_map = relations_map
            self.objects_map = objects_map
            self.subjects_size = subjects_size
            self.relations_size = relations_size
            self.objects_size = objects_size
        else:
            assert base_filename
            subjects_map, relations_map, objects_map, subjects_size, relations_size, objects_size = self.get_dict(
                base_filename)
            self.subjects_map = subjects_map
            self.relations_map = relations_map
            self.subjects_size = subjects_size
            self.relations_size = relations_size

    @staticmethod
    def get_dict(filename):
        subjects_map, s_idx = defaultdict(int), 1
        relations_map, r_idx = defaultdict(int), 1
        objects_map, o_idx = defaultdict(int), 1
        with open(filename) as f:
            for line in tqdm(f, desc="build dict from file[%s]" % filename):
                if not line.strip():
                    continue
                rdf_triple = line.split()
                assert len(rdf_triple) == 3
                sub, rel, obj = rdf_triple
                if sub not in subjects_map:
                    subjects_map[sub] = s_idx
                    s_idx += 1
                if obj not in subjects_map:
                    objects_map[obj] = o_idx
                    o_idx += 1
                if rel not in relations_map:
                    relations_map[rel] = r_idx
                    r_idx += 1
        return subjects_map, relations_map, objects_map, s_idx, r_idx, o_idx

    def transform(self, source, target):
        wf = wf_open(target)
        with open(source) as f:
            for line in tqdm(f, "%s transform %s to %s" % (self.__class__.__name__, source, target)):
                line = line.strip()
                if line:
                    rdf_triple = line.split()
                    assert len(rdf_triple) == 3
                    print("%s\t%s\t%s" % (self.subjects_map[rdf_triple[0]], self.relations_map[rdf_triple[1]],
                                          self.objects_map[rdf_triple[2]]), file=wf)
        wf_close(wf)

    def save(self, subjects_map_filename, relations_map_filename, objects_map_filename):
        wf = wf_open(subjects_map_filename)
        json.dump(self.subjects_map, wf, ensure_ascii=False)
        wf_close(wf)
        wf = wf_open(relations_map_filename)
        json.dump(self.relations_map, wf, ensure_ascii=False)
        wf_close(wf)
        wf = wf_open(relations_map_filename)
        json.dump(self.objects_map, wf, ensure_ascii=False)
        wf_close(wf)

    @staticmethod
    def load(subjects_map_filename, relations_map_filename, objects_map_filename):
        with open(subjects_map_filename) as f:
            subjects_map = defaultdict(int)
            subjects_map.update(json.load(f))
            entities_size = len(subjects_map) + 1
        with open(relations_map_filename) as f:
            relations_map = defaultdict(int)
            relations_map.update(json.load(f))
            relations_size = len(relations_map) + 1

        with open(objects_map_filename) as f:
            objects_map = defaultdict(int)
            objects_map.update(json.load(f))
            objects_size = len(objects_map) + 1

        return subjects_map, relations_map, objects_map, entities_size, relations_size, objects_size


class UniteMapper(Mapper):
    def __init__(self, unite_map=None, unite_size=None, base_filename=None):
        if unite_map and unite_size:
            self.unite_map = unite_map
            self.unite_size = unite_size
        else:
            assert base_filename
            unite_map, unite_size = self.get_dict(base_filename)
            self.unite_map = unite_map
            self.unite_size = unite_size

    @staticmethod
    def get_dict(filename):
        unite_map, u_idx = defaultdict(int), 1
        with open(filename) as f:
            for line in tqdm(f, desc="build dict from file[%s]" % filename):
                if not line.strip():
                    continue
                rdf_triple = line.split()
                assert len(rdf_triple) == 3
                sub, rel, obj = rdf_triple
                if sub not in unite_map:
                    unite_map[sub] = u_idx
                    u_idx += 1
                if obj not in unite_map:
                    unite_map[obj] = u_idx
                    u_idx += 1
                if rel not in unite_map:
                    unite_map[rel] = u_idx
                    u_idx += 1
        return unite_map, u_idx

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

    def save(self, unite_map_filename):
        wf = wf_open(unite_map_filename)
        json.dump(self.unite_map, wf, ensure_ascii=False)
        wf_close(wf)

    @staticmethod
    def load(unite_map_filename):
        with open(unite_map_filename) as f:
            unite_map = defaultdict(int)
            unite_map.update(json.load(f))
            unite_size = len(unite_map) + 1
        return unite_map, unite_size
