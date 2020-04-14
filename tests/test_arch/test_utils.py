# coding: utf-8
# 2020/4/14 @ tongshiwei

from longling import as_out_io, path_append
from longling.lib.testing import simulate_stdin
from longling.Architecture.install_file import copytree, copyfile
from longling.Architecture import config


def test_copy(tmp_path):
    src_dir = path_append(tmp_path, "src")
    tar_dir = path_append(tmp_path, "tar")

    src = path_append(src_dir, "src.txt")
    tar = path_append(tar_dir, "tar.txt")

    with as_out_io(src) as wf:
        print("hello world", file=wf)

    config.OVERRIDE = False
    copytree(src_dir, tar_dir)
    copytree(src_dir, tar_dir)
    copyfile(src, tar)

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
