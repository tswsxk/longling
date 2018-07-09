# coding:utf-8
# created by tongshiwei on 2018/7/9

import random
import math
import json

from tqdm import tqdm

from longling.lib.stream import wf_open, wf_close
from longling.framework.KG.io_lib import load_plain, rdf2sro, rdf2rso


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
    sro, entities, _ = rdf2sro(load_plain(source))

    wf = wf_open(loc_jsonxz)
    for s, ro in tqdm(sro.items(), source):
        for r, o in ro.items():
            nos = random.sample(entities - o, len(o))
            for i, obj in enumerate(o):
                print(json.dumps({"x": (s, r, obj), "z": (s, r, nos[i])}, ensure_ascii=False), file=wf)
    wf_close(wf)


