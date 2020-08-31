# coding: utf-8
# create by tongshiwei on 2019/6/27

from pathlib import PurePath
import sys
import json
import pickle
import pytest
from longling import path_append, tmpfile
from longling.lib.stream import *


def test_flush_print(capsys):
    flush_print("test flush print")
    captured = capsys.readouterr()
    assert captured.out == "\rtest flush print"


def test_write(capsys):
    with as_out_io() as wf:
        print("hello world", file=wf)

    captured = capsys.readouterr()

    assert captured.out == "hello world\n"

    wf = wf_open(mode="stdout")
    print("test stdout", file=wf)
    close_io(wf)

    captured = capsys.readouterr()

    assert captured.out == "test stdout\n"

    wf = wf_open(mode="stderr")
    print("test stderr", file=wf)
    close_io(wf)

    captured = capsys.readouterr()

    assert captured.err == "test stderr\n"

    with pytest.raises(TypeError):
        wf_open(mode="error")

    assert not check_file("error")


def test_read_write(tmp_path):
    tmp_file = path_append(tmp_path, "read_write")

    with as_io() as f:
        assert f == sys.stdin

    with as_out_io(tmp_file) as wf:
        print("test read write 1", file=wf)

    with as_out_io(str(tmp_file), mode="a") as wf:
        print("test read write 2", file=wf)

    _sum = 0
    with as_io([tmp_file]) as f:
        for line in f:
            if line.strip():
                _sum += int(line.strip().split(" ")[-1])

    assert _sum == 3

    tmp_file = path_append(tmp_path, "tmp_path", "tmp_file")

    with as_out_io(tmp_file, mode="wb") as wf:
        pickle.dump({"test": 123}, wf)

    assert pickle_load(tmp_file)["test"] == 123

    with as_io(tmp_file, mode="rb") as f:
        assert pickle_load(f)["test"] == 123

    with as_out_io(tmp_file) as wf:
        json.dump({"test": 123}, wf)

    assert json_load(tmp_file)["test"] == 123

    with as_out_io(tmp_file) as wf:
        print(json.dumps({"test": 123}), file=wf)

    assert json_load(tmp_file)["test"] == 123

    assert check_file(tmp_file)

    with tmpfile("manual") as tmp:
        with as_out_io(tmp, encoding=None) as wf:
            print("hello world", file=wf)

        with as_io([tmp], encoding=None) as f:
            for line in f:
                assert line == "hello world\n"

    # Exception Test
    with pytest.raises(ValueError):
        with as_io(tmp_file, mode="unk"):
            pass

    # Exception Test
    with pytest.raises(TypeError):
        wf_open({"123": 123})

    with pytest.raises(StreamError):
        close_io("123")


def test_group_io(tmpdir):
    filenames = [path_append(tmpdir, "%s.txt" % i) for i in range(4)]

    with as_out_io_group(*filenames) as wfs:
        for i in range(4):
            print(i, file=wfs[i])

    with as_io_group(*filenames) as fs:
        for i in range(4):
            assert "%s" % i == fs[i].readline().strip()

    wfs = wf_group_open(*filenames)
    for i in range(4):
        print(i, file=wfs[i])
    close_io(wfs)

    fs = rf_group_open(*filenames)
    for i in range(4):
        assert "%s" % i == fs[i].readline().strip()
    close_io(fs)


def test_io_type(tmp_path):
    with pytest.raises(TypeError):
        wf_open(12345)

    with pytest.raises(TypeError):
        rf_open(12345)

    with as_out_io(path_append(tmp_path, "test_out")) as wf:
        with as_out_io(wf):
            pass

    with as_io(path_append(tmp_path, "test_out")) as f:
        with rf_open(f):
            pass


def test_add(capsys, tmp_path):
    tmp_file = path_append(tmp_path, "test_add", to_str=True)

    wf = wf_open(tmp_file)

    output_flow = AddPrinter(wf)

    assert PurePath(output_flow.name).name == "test_add"

    for i in range(10):
        output_flow.add(i)

    close_io(wf)

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

    output_flow.close()

    with AddPrinter(tmp_file) as a:
        for i in range(10):
            a.add(i)

    with open(tmp_file) as f:
        for i, line in enumerate(f):
            if line.strip():
                assert int(line.strip()) == i


def test_encode(tmpdir):
    demo_text = "测试用中文\nhello world\n如果再重来"

    src = path_append(tmpdir, "gbk.txt")
    tar = path_append(tmpdir, "utf8.txt")

    with wf_open(src, encoding="gbk") as wf:
        print(demo_text, end='', file=wf)

    encode(src, "gbk", tar, "utf-8")

    with rf_open(tar) as f:
        for line in f:
            print(line)
