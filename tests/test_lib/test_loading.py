# coding: utf-8
# 2020/1/2 @ tongshiwei

from longling.lib.loading import load_csv, load_json, json2csv, csv2json, loading
from longling import path_append, wf_open

DEMO_TEXT = """
name,id
Tom,0
Jerry,1
"""


def text_to_csv(path):
    src = path_append(path)
    with wf_open(src) as wf:
        print(DEMO_TEXT.strip(), file=wf)


def test_load_csv(tmpdir):
    src = path_append(tmpdir, "test.csv")
    text_to_csv(src)

    for i, line in enumerate(load_csv(src)):
        assert int(line["id"]) == i
        if i == 0:
            assert line["name"] == "Tom"
        elif i == 1:
            assert line["name"] == "Jerry"


def test_load_json(tmpdir):
    csv_src = path_append(tmpdir, "test.csv")
    src = path_append(tmpdir, "json.csv")
    text_to_csv(csv_src)

    csv2json(csv_src, src)

    for i, line in enumerate(load_json(src)):
        assert int(line["id"]) == i
        if i == 0:
            assert line["name"] == "Tom"
        elif i == 1:
            assert line["name"] == "Jerry"


def test_loading(tmpdir):
    csv_src = path_append(tmpdir, "test.csv")
    json_src = path_append(tmpdir, "test.json")

    text_to_csv(csv_src)
    csv2json(csv_src, json_src)
    json2csv(json_src, csv_src)

    for src in [csv_src, json_src, load_json(json_src)]:
        for i, line in enumerate(loading(src)):
            assert int(line["id"]) == i, line
            if i == 0:
                assert line["name"] == "Tom", line
            elif i == 1:
                assert line["name"] == "Jerry", line
