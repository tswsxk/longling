# coding: utf-8
# create by tongshiwei on 2018/7/8
"""
提供不同的vec文件的转换方法
格式说明见 ../README.md
"""
import json

from tqdm import tqdm

from longling.lib.stream import wf_open, wf_close


def data2tup(tuples, loc_tup):
    """

    Parameters
    ----------
    tuples
    loc_tup

    Returns
    -------

    """
    wf = wf_open(loc_tup)
    for tup in tqdm(tuples):
        print(json.dumps(tup, ensure_ascii=False), file=wf)
    wf_close(wf)


def data2dat(tuples, loc_dat):
    """

    Parameters
    ----------
    tuples
    loc_dat

    Returns
    -------

    """
    wf = wf_open(loc_dat)
    for tup in tqdm(tuples):
        print(r"%s %s" % (tup[0], " ".join([str(t) for t in tup[1]])), file=wf)
    wf_close(wf)


def tup2dat(loc_tup, loc_dat):
    """
    convert .vec.dup type file to type .vec.dat
    Parameters
    ----------
    loc_tup
    loc_dat

    Returns
    -------

    """
    wf = wf_open(loc_dat)
    with open(loc_tup) as f:
        for line in tqdm(f):
            data = json.loads(line)
            print(r"%s %s" % (data[0], " ".join([str(d) for d in data[1]])), file=wf)
    wf_close(wf)


def dat2tup(loc_dat, loc_tup):
    """
    convert .vec.dat type file to type .vec.tup
    Parameters
    ----------
    loc_dat: str
    loc_tup: str

    Returns
    -------

    """
    wf = wf_open(loc_tup)
    with open(loc_dat) as f:
        for line in tqdm(f):
            data = line.split()
            print(json.dumps((data[0], [float(d) for d in data[1:]])), file=wf)
    wf_close(wf)


def __load(loc_file, line_parse):
    embedding_dict = dict()
    with open(loc_file) as f:
        for line in tqdm(f):
            key, value = line_parse(line)
            embedding_dict[key] = value
    return embedding_dict


def __tup_line_parse(line):
    datas = json.loads(line)
    return datas[0], datas[1]


def __dat_line_parse(line):
    datas = line.strip()
    return datas[0], [float(d) for d in datas[1:]]


def load_tup(loc_tup):
    return __load(loc_tup, __tup_line_parse)


def load_dat(loc_dat):
    return __load(loc_dat, __dat_line_parse)
