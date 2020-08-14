# coding: utf-8
# 2019/9/20 @ tongshiwei

__all__ = ["cli"]

import os

from longling.Architecture.cli.units import *
from longling.Architecture.install_proj import project_types
from longling.Architecture.utils import legal_input, binary_legal_input, default_legal_input
from longling.lib.path import path_append

from .. import config


def cli(skip_top=True, project=None, override=None, tar_dir="./", **kwargs):
    """
    The main function for arch
    """
    config.OVERRIDE = override

    main_params = main_cli(skip_top, project, **kwargs)

    kwargs.update(main_params)

    indicator = dict(
        travis=binary_legal_input("Install travis?"),
    )

    docs = binary_legal_input("Install docs?")
    if main_params["project_type"] == "docs" or docs:
        default_style = "mxnet" if main_params["project_type"] == "docs" else "sphinx"
        docs_params = docs_cli(default_style=default_style, **kwargs)
    else:
        docs_params = {}

    service_params = {}
    docker_params = {}
    if main_params["project_type"] != "docs":
        service = binary_legal_input("To deploy as a service")
        if service:
            docker_params.update(
                dockerfile_cli(project_type=main_params["project_type"])
            )
            if "port" not in docker_params:
                port = default_legal_input("Image Port", __default_value=None)
            else:
                port = docker_params["port"]
            service_params["port"] = None if port == 'null' else port
            service_params.update(dict(
                private=binary_legal_input("Is private project?"),
            ))
            stages_candidates = {
                "test": {"stage_image_name": docker_params["image_name"]}
            }
            if main_params["project_type"] == "python":
                stages_candidates["build"] = {"need": "n"}
            if binary_legal_input("Install .gitlab-ci.yml?"):
                service_params.update(dict(
                    gitlab_ci_params=gitlab_ci_cli(
                        port=port, docs=docs,
                        stages_candidates=stages_candidates
                    ),
                ))

        elif binary_legal_input("Install Dockerfile?"):
            docker_params.update(dockerfile_cli(project_type=main_params["project_type"]))

    if skip_top:
        tar_dir = path_append(tar_dir, "./")
    else:
        tar_dir = path_append(tar_dir, main_params["project"])
        os.makedirs(tar_dir)

    __project_type = main_params["project_type"]

    project_types[__project_type](
        tar_dir=tar_dir, main_params=main_params, docs_params=docs_params,
        docker_params=docker_params, service_params=service_params, **indicator
    )


if __name__ == '__main__':
    cli()
