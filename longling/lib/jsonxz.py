# coding:utf-8
# created by tongshiwei on 2018/7/25

import json
from json import loads, dumps

from tqdm import tqdm


def dump(objs, fp, wrapper=tqdm):
    for obj in wrapper(objs):
        print(json.dumps(obj, ensure_ascii=False), file=fp)


def verify(filename):
    with open(filename) as f:
        for line in tqdm(f):
            loads(line)
