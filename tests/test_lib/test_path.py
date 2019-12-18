# coding: utf-8
# 2019/12/11 @ tongshiwei

from longling import path_append, wf_open
from longling import file_exist, abs_current_dir


def test_path(tmp_path):
    path_append(tmp_path, "../data", "../dataset1/", "train", to_str=True)

    tmp_file = path_append(tmp_path, "test_path.txt")
    with wf_open(tmp_file) as wf:
        print("hello world", file=wf)

    assert file_exist(tmp_file)
    abs_current_dir(tmp_file)
