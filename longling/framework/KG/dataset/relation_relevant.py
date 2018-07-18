# coding: utf-8
# create by tongshiwei on 2018/7/18
from __future__ import division

import random

from tqdm import tqdm

from longling.lib.stream import wf_open

from longling.framework.KG.base import logger
from longling.framework.KG.io_lib import load_plain, rdf2rso, rdf2ros


class KGRelationError(Exception):
    pass


def relation_classification(filename, target_prefix, threshold=1.5):
    """
    to build 1-1, 1-n, n-1, n-n dataset
    Parameters
    ----------
    filename
    target_prefix
    threshold

    Returns
    -------

    """
    one2one_filename = target_prefix + "_1to1"
    one2many_filename = target_prefix + "_1ton"
    many2one_filename = target_prefix + "_nto1"
    many2many_filename = target_prefix + "_n2n"

    one2one = []
    one2many = []
    many2one = []
    many2many = []

    rdf_triples = load_plain(filename)
    rso = rdf2rso(rdf_triples)
    ros = rdf2ros(rdf_triples)

    for rel in rso:
        sub2nobj = sum([len(objs) for objs in rso[rel].values]) / sum(rso[rel].keys())
        obj2nsub = sum([len(subs) for subs in ros[rel].values]) / sum(ros[rel].keys())
        if sub2nobj < threshold and obj2nsub < threshold:
            one2one.append(rel)
        elif obj2nsub >= threshold:
            many2one.append(rel)
        elif obj2nsub < threshold:
            one2many.append(rel)
        else:
            many2many.append(rel)

    one2one = set(one2one)
    one2many = set(one2many)
    many2one = set(many2one)
    many2many = set(many2many)

    with open(filename) as f, wf_open(one2one_filename) as one2one_wf, wf_open(
            one2many_filename) as one2many_wf, wf_open(many2one_filename) as many2one_wf, \
            wf_open(many2many_filename) as many2many_wf:
        for i, line in tqdm(enumerate(f), "relation classification for %s" % filename):
            rdf_triple = rdf_triples[i]
            if rdf_triple[1] in one2one:
                print(line, end="", file=one2one_wf)
            elif rdf_triple[1] in one2many:
                print(line, end="", file=one2many_wf)
            elif rdf_triple[1] in many2one:
                print(line, end="", file=many2one_wf)
            elif rdf_triple[1] in many2many:
                print(line, end="", file=many2many_wf)
            else:
                logger.warn("rel not found in any relation set, may caused extremely serious error: line %s\n%s", i,
                            line)


def relation_prediction(source, n_rel_filename_prefix, n_rest_filename_prefix, rel_num=40, rel_triples_num=1000):
    train_suffix = "_train"
    test_suffix = "_test"

    n_rel_train = n_rel_filename_prefix + train_suffix
    n_rel_test = n_rel_filename_prefix + test_suffix

    n_rest_train = n_rest_filename_prefix + train_suffix
    n_rest_test = n_rest_filename_prefix + test_suffix

    rdf_triples = load_plain(source)
    rso = rdf2rso(rdf_triples)
    n_rel_index = set(random.sample(range(len(rso)), rel_num))
    with wf_open(n_rel_train) as n_rel_train_wf, wf_open(n_rel_test) as n_rel_test_wf, wf_open(
            n_rest_train) as n_rest_train_wf, wf_open(n_rest_test) as n_rest_test:
        for idx, rel in tqdm(enumerate(rso)):
            if idx in n_rel_index:
                samples = []
                for sub, objs in rso[rel].items():
                    samples.extend([(sub, obj) for obj in objs])
                samples = random.shuffle(samples)
                train_samples = samples[:rel_triples_num]
                for sample in train_samples:
                    print()
                test_samples = samples[rel_triples_num:]
            else:
                pass
