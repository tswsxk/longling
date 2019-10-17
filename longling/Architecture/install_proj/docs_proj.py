# coding: utf-8
# 2019/9/21 @ tongshiwei

from longling.lib.stream import build_dir
from longling.lib.path import path_append
from longling.Architecture.install_file import sphinx_conf, readthedocs, gitignore


def docs_proj(tar_dir, docs_params, __gitignore=True, **kwargs):
    assert docs_params
    variables = {}
    variables.update(docs_params)

    docs_root = build_dir(path_append(tar_dir, docs_params["docs_root"]))
    sphinx_conf(tar_dir=docs_root, **docs_params)

    readthedocs(tar_dir=tar_dir)

    if __gitignore:
        gitignore("docs", tar_dir)
