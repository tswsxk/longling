# coding: utf-8
# 2019/9/20 @ tongshiwei

from longling.Architecture.install_proj.docs_proj import docs_proj
from longling.Architecture.install_file import pysetup, makefile, pytest, coverage
from longling.Architecture.install_file import gitignore, gitlab_ci, dockerfile, chart, travis


def py_proj(tar_dir, main_params, docs_params, docker_params=None, service_params=None, **kwargs):
    variables = {}
    variables.update(main_params)

    pysetup(tar_dir, **variables)

    makefile(tar_dir, **variables)

    pytest(tar_dir)
    coverage(tar_dir, **variables)

    gitignore("python", tar_dir)

    if docs_params:
        docs_proj(tar_dir, docs_params, __gitignore=False)

    if kwargs.get("travis") is True:
        travis(tar_dir)

    if docker_params:
        dockerfile("python-%s" % docker_params["docker_type"], tar_dir=tar_dir, **docker_params)

    if service_params:
        if service_params.get("gitlab_ci_params"):
            chart(tar_dir)
            gitlab_ci(
                atype=variables["project_type"],
                stages=service_params["gitlab_ci_params"],
                private=service_params["private"],
                tar_dir=tar_dir
            )
