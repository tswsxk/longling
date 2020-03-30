# coding: utf-8
# 2020/1/2 @ tongshiwei

from longling.lib.loading import load_csv, load_jsonl, json2csv, csv2json, loading
from longling import path_append, as_out_io, as_io

DEMO_TEXT = """
name,id
Tom,0
Jerry,1
"""


def text_to_csv(path):
    with wf_open(path) as wf:
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

    for i, line in enumerate(load_jsonl(src)):
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

    for src in [csv_src, json_src, load_jsonl(json_src)]:
        for i, line in enumerate(loading(src)):
            assert int(line["id"]) == i, line
            if i == 0:
                assert line["name"] == "Tom", line
            elif i == 1:
                assert line["name"] == "Jerry", line

    src = path_append(tmpdir, "test")
    with as_out_io(src) as wf:
        print(DEMO_TEXT.strip(), file=wf)

    assert [line.strip() for line in loading(src)] == DEMO_TEXT.strip().split("\n")
    with as_io(src) as f:
        assert [line.strip() for line in loading(f)] == DEMO_TEXT.strip().split("\n")
    assert "hello world" == loading(lambda: "hello world")


