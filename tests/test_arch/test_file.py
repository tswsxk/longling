# coding: utf-8
# 2019/12/31 @ tongshiwei

from longling import wf_open, path_append
from longling.Architecture.install_file import template_copy
from longling.Architecture.install_file import gitignore, pytest as gen_pytest, coverage, pysetup, sphinx_conf
from longling.Architecture.install_file import readthedocs, makefile


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


def test_files(tmpdir):
    # test gitignore
    for atype in ["", "docs", "python"]:
        gitignore(atype=atype, tar_dir=tmpdir)

    gen_pytest()
    coverage(tmpdir, project="longling")

    with open(path_append(tmpdir, "setup.cfg")) as f:
        f.readline()
        assert f.readline().strip() == "source=longling"

    pysetup(tmpdir, project="longling")
    sphinx_conf(tmpdir)
    readthedocs(tmpdir)
    makefile(tmpdir, project="longling")

    with open(path_append(tmpdir, "Makefile")) as f:
        assert f.readline().strip() == r"""VERSION=`ls dist/*.tar.gz | sed "s/dist\/longling-\(.*\)\.tar\.gz/\1/g"`"""
