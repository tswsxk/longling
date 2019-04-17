# coding: utf-8
# create by tongshiwei on 2018/7/18
from tqdm import tqdm


def load_plain(rdf_txt, to_int=False):
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
                if to_int:
                    rdf_triple = [int(elem) for elem in rdf_triple]
                rdf_triples.append(rdf_triple)
    return rdf_triples


def load_plains(rdf_txt_files, to_int=False):
    rdf_triples = []
    for rdf_txt in rdf_txt_files:
        rdf_triples.extend(load_plain(rdf_txt, to_int))
    return rdf_triples
