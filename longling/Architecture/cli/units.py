# coding: utf-8
# 2019/9/21 @ tongshiwei

__all__ = ["main_cli", "docs_cli", "dockerfile_cli", "gitlab_ci_cli"]

import datetime
import os
import pathlib
from collections import OrderedDict
from copy import deepcopy

from longling.Architecture.config import STAGE_CANDIDATES
from longling.Architecture.install_proj import project_types
from longling.Architecture.utils import legal_input, binary_legal_input, default_legal_input


def main_cli(skip_top, project, **kwargs):
    if skip_top:
        default_project_name = pathlib.PurePath(os.getcwd()).name
        project = legal_input("Project Name (default is %s) < " % default_project_name,
                              __default_value="%s" % default_project_name) if not project else project
    else:
        project = legal_input("Project Name < ", is_legal=lambda x: True if x else False) if not project else project
    params = dict(
        project=legal_input("Project Name < ", is_legal=lambda x: True if x else False) if not project else project,
        project_type=legal_input("Project Type (%s) < " % "/".join(project_types), __legal_input=project_types),
    )
    return params


def docs_cli(project=None, title=None, author=None, copyright=None, default_style="sphinx",
             **kwargs):
    params = dict(
        project=legal_input("Project Name < ", is_legal=lambda x: True if x else False) if not project else project,
        docs_style=default_legal_input("Docs Style",
                                       __legal_input={"mxnet", "sphinx"},
                                       __default_value="%s" % default_style),
        docs_root="docs/" if binary_legal_input("Make 'docs/' directory?", _default="y") else "./",
    )
    if params["docs_style"] != "sphinx":
        params.update(dict(
            title=legal_input("Docs Title (default is %s) < " % params["project"],
                              __default_value=params["project"]) if not title else title,
            author=legal_input("Author < ", is_legal=lambda x: True if x else False) if not author else author,

        ))

        default_copyright = "%s, %s" % (datetime.date.today().year, params["author"])
        params.update(dict(
            copyright=legal_input("Copyright (default is %s) < " % default_copyright,
                                  __default_value=default_copyright) if not copyright else copyright,
        ))
    params.update(readthedocs=binary_legal_input("Install .readthedocs.yml?"))

    if binary_legal_input("Install Dockerfile for documents?"):
        params.update(dict(
            docker_params=dockerfile_cli("docs", docker_type="nginx"))
        )

    return params


def dockerfile_cli(project_type, docker_type=None, port=None):
    project_docker = {
        "python": {"cli", "flask"},
        "web": {"nginx", "vue"},
        "docs": {"nginx"},
    }

    docker_params = {}

    docker_type = legal_input(
        "Choose a service type (%s) < " % "/".join(project_docker[project_type]),
        __legal_input=project_docker[project_type],
    ) if docker_type is None else docker_type

    docker_params["docker_type"] = docker_type

    if docker_type in {"nginx", "vue"}:
        docker_params.update(
            image_name=default_legal_input(
                "Choose a image", __default_value="nginx"
            ),
            path_to_html=default_legal_input(
                "Specify the html directory",
                __default_value="_build/html" if project_type == "docs" else "build/html"
            )
        )

    if project_type == "python":
        docker_params.update(
            image_name=default_legal_input(
                "Choose a image", __default_value="python:3.6"
            )
        )
        if docker_type == "cli":
            docker_params.update(
                path_to_main=default_legal_input(
                    "Specify the main entry (e.g., the path to main.py)",
                    is_legal=lambda x: True if x else False
                )
            )
        elif docker_type == "flask":
            docker_params.update(
                path_to_main=default_legal_input(
                    "Specify the main entry (e.g., main_package.main_py:main_func)",
                    is_legal=lambda x: True if x else False
                ),
                port=default_legal_input(
                    "Specify the port that docker will listen",
                    is_legal=lambda x: True if x else False
                ) if port is None else port
            )

        else:  # pragma: no cover
            raise TypeError("%s: cannot handle docker type %s, only supports %s" % (
                project_type, docker_type, project_docker[project_type]))

    return docker_params


def gitlab_ci_cli(port=None, stages_candidates: (OrderedDict, dict) = None, docs=False, **kwargs):
    def get_stage_params(stage_image_name=None, stage_image_port=None, on_stop=False, only_master=None, manual=None):
        _params = {}
        if stage_image_name is None:
            _params["image_name"] = default_legal_input("Choose a image", is_legal=lambda x: True if x else False)
        else:
            _params["image_name"] = default_legal_input("Choose a image", __default_value=stage_image_name)

        if stage_image_port is not None:
            _params["image_port"] = legal_input("Stage Image Port (default is %s) < " % stage_image_port,
                                                __default_value=stage_image_port)
        if on_stop:
            _params["on_stop"] = binary_legal_input("Add Corresponding Stop Stage?")
        if only_master is not None:
            _params["only_master"] = binary_legal_input("Only triggered in master branch?", _default=only_master)
        if manual is not None:
            _params["manual"] = binary_legal_input("Triggered manually?", _default=manual)
        return _params

    _stages_candidates = deepcopy(STAGE_CANDIDATES)

    if stages_candidates is not None:
        _stages_candidates.update(stages_candidates)

    if docs is False:
        _stages_candidates.pop("docs")

    deployment_image = None

    params = {}
    for stage, ques_params in _stages_candidates.items():
        if binary_legal_input("Need [%s] Stage?" % stage, _default=ques_params.get("need", "y")):
            if stage in {"review", "production"}:
                if "image_port" not in ques_params and port is not None:
                    ques_params.update({"stage_image_port": port})
                if "stage_image_name" not in ques_params and deployment_image is not None:
                    ques_params.update({"stage_image_name": deployment_image})
            params[stage] = get_stage_params(**ques_params)
            if stage in {"review", "production"} and "image_name" in params[stage]:
                deployment_image = params[stage]["image_name"]
            elif stage == "build" and "image_name" in params[stage]:
                _stages_candidates["test"]["stage_image_name"] = params[stage]["image_name"]

    return params
