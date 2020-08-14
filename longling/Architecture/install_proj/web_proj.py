# coding: utf-8
# 2020/3/3 @ tongshiwei

from longling.Architecture.install_proj.docs_proj import docs_proj
from longling.Architecture.install_file import gitignore, travis, dockerfile, gitlab_ci, chart


def web_proj(tar_dir, main_params, docs_params, docker_params=None, service_params=None, **kwargs):
    variables = {}
    variables.update(main_params)

    gitignore("web", tar_dir)

    if docs_params:
        docs_proj(tar_dir, docs_params, __gitignore=False)

    if kwargs.get("travis") is True:
        travis(tar_dir)

    if docker_params:
        dockerfile("nginx", tar_dir=tar_dir, **docker_params)

    if service_params:
        if service_params.get("gitlab_ci_params"):
            chart(tar_dir)
            gitlab_ci(
                atype=docker_params["docker_type"],
                stages=service_params["gitlab_ci_params"],
                private=service_params["private"],
                tar_dir=tar_dir,
                version_in_path=False if docker_params["docker_type"] == "vue" else True,
            )
