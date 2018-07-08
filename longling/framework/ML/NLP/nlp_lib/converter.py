# coding: utf-8
# create by tongshiwei on 2018/7/8

from longling.lib.stream import wf_open, wf_close

def tup2dat(loc_tup, loc_dat):
    wf = wf_open(loc_dat)
    with open(loc_tup) as f: