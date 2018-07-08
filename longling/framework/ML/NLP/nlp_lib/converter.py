# coding: utf-8
# create by tongshiwei on 2018/7/8

import json

from tqdm import tqdm

from longling.lib.stream import wf_open, wf_close


def tup2dat(loc_tup, loc_dat):
    wf = wf_open(loc_dat)
    with open(loc_tup) as f:
        for line in tqdm(f):
            data = json.loads(line)
            print("%s %s" % (data[0], " ".join([str(d) for d in data[1]])), file=wf)
    wf_close(wf)


def dat2tup(loc_dat, loc_tup):
    wf = wf_open(loc_tup)
    with open(loc_dat) as f:
        for line in tqdm(f):
            data = line.split()
            print(json.dumps((data[0], [float(d) for d in data[1:]])), file=wf)
    wf_close(wf)
