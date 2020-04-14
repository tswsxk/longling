# coding: utf-8
# 2019/9/21 @ tongshiwei

import os
from longling.lib.path import path_append
from longling.Architecture.install_file import sphinx_conf, readthedocs, gitignore, dockerfile


def docs_proj(tar_dir, docs_params, docker_params=None, __gitignore=True, **kwargs):
    assert docs_params

    variables = {}
    variables.update(docs_params)

    docs_root = path_append(tar_dir, docs_params["docs_root"])
    if not os.path.exists(docs_root):
        os.makedirs(docs_root)
    sphinx_conf(tar_dir=docs_root, **docs_params)

    if docs_params["readthedocs"]:
        readthedocs(tar_dir=tar_dir)

    if __gitignore:
        gitignore("docs", tar_dir)

    if docker_params:  # pragma: no cover
        dockerfile(docs_params["docker_type"], **docker_params, tar_dir=docs_root)

    elif "docker_params" in docs_params:
        dockerfile(docs_params["docker_params"]["docker_type"], **docs_params["docker_params"], tar_dir=docs_root)
