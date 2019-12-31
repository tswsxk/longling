# coding: utf-8
# 2019/12/31 @ tongshiwei

from longling import wf_open, path_append
from longling.Architecture.install_file import template_copy


def test_template_copy(tmpdir):
    pseudo_template = """
    project=$PROJECT
    author=$AUTHOR
    """.lstrip()
    src = path_append(tmpdir, "src.template")
    tar = path_append(tmpdir, "tar")
    with wf_open(src) as wf:
        print(pseudo_template, file=wf)

    template_copy(src, tar, quotation='', project="longling", author="sherlock")

    with open(tar) as f:
        assert f.readline().strip() == "project=longling"
        assert f.readline().strip() == "author=sherlock"
