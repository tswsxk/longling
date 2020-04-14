# coding: utf-8
# 2020/4/14 @ tongshiwei

from longling import as_out_io, path_append
from longling.lib.testing import simulate_stdin
from longling.Architecture.install_file import copytree, copyfile, template_copy
from longling.Architecture import config
from longling.Architecture.utils import default_legal_input


def test_copy(tmpdir):
    src_dir = path_append(tmpdir, "src")
    tar_dir = path_append(tmpdir, "tar")

    src = path_append(src_dir, "src.txt")
    tar = path_append(tar_dir, "tar.txt")

    with as_out_io(src) as wf:
        print("hello world", file=wf)

    config.OVERRIDE = False
    copytree(src_dir, tar_dir)
    copytree(src_dir, tar_dir)
    copyfile(src, tar)
    template_copy(src, tar)

    config.OVERRIDE = True
    copytree(src_dir, tar_dir)
    copyfile(src, tar)

    config.OVERRIDE = None
    with simulate_stdin("y", "y"):
        copytree(src_dir, tar_dir)
        copyfile(src, tar)

    with simulate_stdin("n", "n"):
        copytree(src_dir, tar_dir)
        copyfile(src, tar)

    with simulate_stdin("unk", "y"):
        default_legal_input("", __legal_input={"y"})
