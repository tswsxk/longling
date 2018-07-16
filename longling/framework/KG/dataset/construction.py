# coding:utf-8
# created by tongshiwei on 2018/7/9

import random
import math
import json

from tqdm import tqdm

from longling.lib.stream import wf_open, wf_close
from longling.framework.KG.io_lib import load_plain, load_plains, rdf2sro, rdf2rso, rdf2ors


def sro_jsonxz(source, loc_jsonxz, negtive_ratio=1.0):
    sro, entities, _ = rdf2sro(load_plain(source))

    wf = wf_open(loc_jsonxz)
    for s, ro in tqdm(sro.items(), loc_jsonxz):
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


def full_jsonxz(source, loc_jsonxz, sources=None, negtive_ratio=None):
    """
    替换test
    file中的头部或尾部构成全量测试集
    输出格式
    {'x': (s, r, o);
    'z': [(s, r, no1), ..., (s, r, nok), (ns1, r, o), ..., (nsm, r, o)]}

    Parameters
    ----------
    test_file: str
    sources: list[str] or None
    loc_jsonxz: str

    Returns
    -------

    """
    sources_datas = load_plains(sources) if sources is not None else None

    source_data = load_plain(source)

    sro, entities, _ = rdf2sro(source_data)

    if sources_datas:
        pos_sro, _, _ = rdf2sro(sources_datas)
        ors, _, _ = rdf2ors(sources_datas)

    else:
        pos_sro = sro
        ors, _, _ = rdf2ors(source_data)

    wf = wf_open(loc_jsonxz)
    for s, ro in tqdm(sro.items(), loc_jsonxz):
        for r, o in ro.items():
            nobjs = entities - pos_sro[s][r]
            for obj in o:
                nsubs = entities - ors[obj][r]
                if negtive_ratio is not None:
                    max_sub_num = math.ceil(negtive_ratio * len(nsubs) / (len(nsubs) + len(nobjs)))
                    sub_num = min(max_sub_num, len(nsubs))
                    obj_num = min(negtive_ratio - sub_num, len(nobjs))

                    nobjs = random.sample(nobjs, obj_num)
                    nsubs = random.sample(nsubs, sub_num)
                if negtive_ratio == 1:
                    print(
                        json.dumps(
                            {'x': (s, r, obj), 'z': ([(s, r, no) for no in nobjs] + [(ns, r, obj) for ns in nsubs])[0]},
                            ensure_ascii=False),
                        file=wf)
                else:
                    print(
                        json.dumps({'x': (s, r, obj), 'z': [(s, r, no) for no in nobjs] + [(ns, r, obj) for ns in nsubs]},
                                   ensure_ascii=False),
                        file=wf)

    wf_close(wf)
