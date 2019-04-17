# coding: utf-8
# create by tongshiwei on 2018/7/18

# Aborted

# import random
# import math
# import json
#
# from tqdm import tqdm
#
# from longling.lib.stream import wf_open, wf_close
# from longling.framework.KG.io_lib import rdf2sro, rdf2ors, rdf2ros, rdf2rso, load_plain, load_plains
#
#
# def sro_jsonxz(source, loc_jsonxz, negtive_ratio=1.0):
#     sro, entities, _ = rdf2sro(load_plain(source))
#
#     wf = wf_open(loc_jsonxz)
#     for s, ro in tqdm(sro.items(), loc_jsonxz):
#         for r, o in ro.items():
#             for obj in o:
#                 print(json.dumps({"x": (s, r, obj), 'z': 1}, ensure_ascii=False), file=wf)
#             neg_n = int(math.ceil(len(o) * negtive_ratio))
#             nos = random.sample(entities - o, neg_n)
#             for neg_obj in nos:
#                 print(json.dumps({"x": (s, r, neg_obj), 'z': 0}, ensure_ascii=False), file=wf)
#     wf_close(wf)
#
#
# def pair_jsonxz(source, loc_jsonxz):
#     full_jsonxz(source, loc_jsonxz, negtive_ratio=1)


# def sr_neg_json(source, loc_json, map_getter, wrapper=lambda x: x):
#     source_data = load_plain(source)
#     entities = wrapper(map_getter(source_data))
#     sro = rdf2sro(source_data)
#     wf = wf_open(loc_json)
#     for s, ro in tqdm(sro.items(), loc_json):
#         for r, o in ro.items():
#             print(s, r, list(entities - sro[s][r]), file=wf)
#     wf_close(wf)
#
#
# def ro_neg_json(source, loc_json, map_getter, wrapper=lambda x: x):
#     source_data = load_plain(source)
#     entities = wrapper(map_getter(source_data))
#     ors = rdf2ors(source_data)
#     wf = wf_open(loc_json)
#     for o, rs in tqdm(ors.items(), loc_json):
#         for r, s in rs.items():
#             print((list(entities - ors[o][r]), r, o), file=wf)
#     wf_close(wf)
