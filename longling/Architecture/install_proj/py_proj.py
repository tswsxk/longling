# coding: utf-8
# 2019/9/20 @ tongshiwei

from longling.Architecture.install_proj.docs_proj import docs_proj
from longling.Architecture.install_file import *


def py_proj(tar_dir, main_params, docs_params):
    variables = {}
    variables.update(main_params)

    pysetup(tar_dir, **variables)

    pytest(tar_dir)

    gitignore("python", tar_dir)

    if docs_params:
        docs_proj(tar_dir, docs_params, __gitignore=False)
