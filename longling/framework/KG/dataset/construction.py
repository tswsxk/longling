# coding:utf-8
# created by tongshiwei on 2018/7/9

import random
import math
import json

from tqdm import tqdm

from longling.lib.stream import wf_open, wf_close
from longling.framework.KG.io_lib import load_plain, rdf2sro, rdf2rso, rdf2ors


def sro_jsonxz(source, loc_jsonxz, negtive_ratio=1.0):
    sro, entities, _ = rdf2sro(load_plain(source))

    wf = wf_open(loc_jsonxz)
    for s, ro in tqdm(sro.items(), source):
        for r, o in ro.items():
            for obj in o:
                print(json.dumps({"x": (s, r, obj), 'z': 1}, ensure_ascii=False), file=wf)
            neg_n = int(math.ceil(len(o) * negtive_ratio))
            nos = random.sample(entities - o, neg_n)
            for neg_obj in nos:
                print(json.dumps({"x": (s, r, neg_obj), 'z': 0}, ensure_ascii=False), file=wf)
    wf_close(wf)


def pair_jsonxz(source, loc_jsonxz):
    full_jsonxz(source, loc_jsonxz, negtive_ratio=1)


def full_jsonxz(source, loc_jsonxz, negtive_ratio=None, max_num=30):
    source_data = load_plain(source)
    sro, entities, _ = rdf2sro(source_data)
    ors, _, _ = rdf2ors(source_data)

    tail_neg_samples = {}
    head_neg_samples = {}

    wf = wf_open(loc_jsonxz)
    for s, ro in tqdm(sro.items(), source):
        for r, o in ro.items():
            if (s, r) not in tail_neg_samples:
                nobjs = entities - o
                num = min(len(nobjs), max_num)
                nobjs = nobjs if len(nobjs) < max_num else random.sample(nobjs, num)
                tail_neg = [(s, r, nobj) for nobj in nobjs]
                tail_neg_samples[(s, r)] = tail_neg
            for obj in o:
                if (r, obj) not in head_neg_samples:
                    nsubs = entities - ors[obj][r]
                    num = min(len(nsubs), max_num)
                    nsubs = nsubs if len(nsubs) < max_num else random.sample(nsubs, num)
                    head_neg = [(nsub, r, obj) for nsub in nsubs]
                    head_neg_samples[(r, obj)] = head_neg
                if negtive_ratio is None:
                    print(json.dumps({'x': (s, r, obj), 'z': head_neg_samples[(r, obj)] + tail_neg_samples[(s, r)]}),
                          file=wf)
                elif negtive_ratio == 1:
                    print(json.dumps({'x': (s, r, obj),
                                      'z': random.sample(head_neg_samples[(r, obj)]
                                                         + tail_neg_samples[(s, r)], 1)[0]}),
                          file=wf)
                else:
                    print(json.dumps({'x': (s, r, obj),
                                      'z': random.sample(head_neg_samples[(r, obj)] + tail_neg_samples[(s, r)],
                                                         negtive_ratio)}),
                          file=wf)
    wf_close(wf)
