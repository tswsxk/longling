# coding:utf-8
# created by tongshiwei on 2018/7/9
from tqdm import tqdm


def rdf2sro(rdf_triples):
    sro = {}
    for rdf_triple in tqdm(rdf_triples):
        if rdf_triple[0] not in sro:
            sro[rdf_triple[0]] = {}
        if rdf_triple[1] not in sro[rdf_triple[0]]:
            sro[rdf_triple[0]][rdf_triple[1]] = set()
        sro[rdf_triple[0]][rdf_triple[1]].add(rdf_triple[2])
    return sro


def rdf2rso(rdf_triples):
    rso = {}
    for rdf_triple in tqdm(rdf_triples):
        if rdf_triple[1] not in rso:
            rso[rdf_triple[1]] = {}
        if rdf_triple[0] not in rso[rdf_triple[1]]:
            rso[rdf_triple[1]][rdf_triple[0]] = set()
        rso[rdf_triple[1]][rdf_triple[0]].add(rdf_triple[2])
    return rso


def rdf2ros(rdf_triples):
    ros = {}
    for rdf_triple in tqdm(rdf_triples):
        if rdf_triple[1] not in ros:
            ros[rdf_triple[1]] = {}
        if rdf_triple[2] not in ros[rdf_triple[1]]:
            ros[rdf_triple[1]][rdf_triple[2]] = set()
        ros[rdf_triple[1]][rdf_triple[2]].add(rdf_triple[0])
    return ros


def rdf2ors(rdf_triples):
    rso = {}
    for rdf_triple in tqdm(rdf_triples):
        if rdf_triple[2] not in rso:
            rso[rdf_triple[2]] = {}
        if rdf_triple[1] not in rso[rdf_triple[2]]:
            rso[rdf_triple[2]][rdf_triple[1]] = set()
        rso[rdf_triple[2]][rdf_triple[1]].add(rdf_triple[0])
    return rso

