# coding: utf-8
# create by tongshiwei on 2019/6/27

import json
import pickle
import pytest
from longling import path_append
from longling.lib.stream import *


def test_flush_print(capsys):
    flush_print("test flush print")
    captured = capsys.readouterr()
    assert captured.out == "\rtest flush print"


def test_write(capsys):
    wf = wf_open(mode="stdout")
    print("test stdout", file=wf)
    wf_close(wf)

    captured = capsys.readouterr()

    assert captured.out == "test stdout\n"

    wf = wf_open(mode="stderr")
    print("test stderr", file=wf)
    wf_close(wf)

    captured = capsys.readouterr()

    assert captured.err == "test stderr\n"

    with pytest.raises(TypeError):
        wf_open(mode="error")

    assert not check_file("error")


def test_read_write(tmp_path):
    tmp_file = path_append(tmp_path, "read_write")

    with wf_open(tmp_file) as wf:
        print("test read write 1", file=wf)

    wf = wf_open(str(tmp_file), mode="a")
    print("test read write 2", file=wf)
    wf_close(wf)

    _sum = 0
    with rf_open(tmp_file) as f:
        for line in f:
            if line.strip():
                _sum += int(line.strip().split(" ")[-1])

    assert _sum == 3

    tmp_file = path_append(tmp_path, "tmp_path", "tmp_file")

    with wf_open(tmp_file, mode="wb") as wf:
        pickle.dump({"test": 123}, wf)

    assert pickle_load(tmp_file)["test"] == 123

    with rf_open(tmp_file, mode="rb") as f:
        assert pickle_load(f)["test"] == 123

    with wf_open(tmp_file) as wf:
        json.dump({"test": 123}, wf)

    assert json_load(tmp_file)["test"] == 123

    with rf_open(tmp_file) as f:
        assert json_load(f)["test"] == 123

    assert check_file(tmp_file)

    # Exception Test
    with pytest.raises(TypeError):
        wf_open({"123": 123})

    with pytest.raises(StreamError):
        wf_close("123")


def test_add(capsys, tmp_path):
    tmp_file = path_append(tmp_path, "test add", to_str=True)

    wf = wf_open(tmp_file)

    output_flow = AddPrinter(wf)

    for i in range(10):
        output_flow.add(i)

    wf_close(wf)

    with open(tmp_file) as f:
        for i, line in enumerate(f):
            if line.strip():
                assert int(line.strip()) == i

    wf = wf_open(mode="stdout")
    output_flow = AddPrinter(wf)

    for i in range(10):
        output_flow.add(i)
        captured = capsys.readouterr()
        assert captured.out == "%s\n" % i

    wf_close(wf)


def test_encoding(tmpdir):
    demo_text = "测试用中文\nhello world\n如果再重来"

    src = path_append(tmpdir, "gbk.txt")
    tar = path_append(tmpdir, "utf8.txt")

    with wf_open(src, encoding="gbk") as wf:
        print(demo_text, end='', file=wf)

    encoding(src, "gbk", tar, "utf-8")

    with rf_open(tar) as f:
        for line in f:
            print(line)
