# coding:utf-8
# created by tongshiwei on 2018/7/9

import json

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


def plain2sro(loc_plain, loc_sro):
    rdf_triples = load_plain(loc_plain)
    sro, _, _ = rdf2sro(rdf_triples)

    with wf_open(loc_sro) as wf:
        for k, v in tqdm(sro.items()):
            print(json.dumps((k, v), ensure_ascii=False), file=wf)



