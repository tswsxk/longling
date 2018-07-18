# coding: utf-8
# create by tongshiwei on 2018/7/18

import json
import math
import random

from tqdm import tqdm

from longling.lib.stream import wf_open, wf_close
from longling.lib.candylib import as_list
from longling.framework.KG.io_lib import load_plain, load_plains, rdf2sro, rdf2ors

from longling.framework.KG.dataset.universe import get_unite, get_er, get_sro


def full_jsonxz(source, loc_jsonxz, sources=None, negative_ratio=None, set_getter=get_er,
                getter_wrapper=lambda x: (x[0], x[1], x[0]), ignore_subject=False, ignore_object=False):
    """
    替换test
    file中的头部或尾部构成测试集
    输出格式
    {'x': (s, r, o);
    'z': [(s, r, no1), ..., (s, r, nok), (ns1, r, o), ..., (nsm, r, o)]}

    Parameters
    ----------
    source: str
    loc_jsonxz: str
    sources: list[str] or None
    negative_ratio: int
    set_getter: get_sro or get_er or get_ubnite
    getter_wrapper:
    ignore_subject: bool
    ignore_object: bool

    Returns
    -------

    """
    assert set_getter in (get_unite, get_er, get_sro)
    sources_datas = load_plains(as_list(sources)) if sources is not None else None

    source_data = load_plain(source)

    subjects, relations, objects = getter_wrapper(set_getter(source_data))

    sro = rdf2sro(source_data)

    if sources_datas:
        pos_sro = rdf2sro(sources_datas)
        ors = rdf2ors(sources_datas)

    else:
        pos_sro = sro
        ors = rdf2ors(source_data)

    wf = wf_open(loc_jsonxz)
    for s, ro in tqdm(sro.items(), loc_jsonxz):
        for r, o in ro.items():
            nobjs = objects - pos_sro[s][r] if not ignore_object else []
            for obj in o:
                nsubs = subjects - ors[obj][r] if not ignore_subject else []
                if not len(nsubs) + len(nobjs) > 0:
                    continue
                if negative_ratio is not None:
                    max_sub_num = math.ceil(negative_ratio * len(nsubs) / (len(nsubs) + len(nobjs)))
                    sub_num = min(max_sub_num, len(nsubs))
                    obj_num = min(negative_ratio - sub_num, len(nobjs))

                    nobjs = random.sample(nobjs, obj_num)
                    nsubs = random.sample(nsubs, sub_num)
                if negative_ratio == 1:
                    print(
                        json.dumps(
                            {'x': (s, r, obj), 'z': ([(s, r, no) for no in nobjs] + [(ns, r, obj) for ns in nsubs])[0]},
                            ensure_ascii=False),
                        file=wf)
                else:
                    print(
                        json.dumps(
                            {'x': (s, r, obj), 'z': [(s, r, no) for no in nobjs] + [(ns, r, obj) for ns in nsubs]},
                            ensure_ascii=False),
                        file=wf)

    wf_close(wf)
