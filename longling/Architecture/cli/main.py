# coding: utf-8
# 2019/9/20 @ tongshiwei

__all__ = ["cli"]

import os
from longling.Architecture.cli.utils import legal_input, binary_legal_input
from longling.Architecture.install_proj import project_types
from longling.Architecture.cli.units import *


def cli(skip_top=True, project=None, **kwargs):  # pragma: no cover
    """
    The main function for arch
    """
    main_params = main_cli(skip_top, project, **kwargs)

    kwargs.update(main_params)

    indicator = dict(
        traivs=binary_legal_input("Install travis?"),
    )

    if main_params["project_type"] == "docs" or binary_legal_input("Install docs?"):
        default_style = "mxnet" if main_params["project_type"] == "docs" else "sphinx"
        docs_params = docs_cli(default_style=default_style, **kwargs)
    else:
        docs_params = {}

    if main_params["project_type"] != "docs":
        if binary_legal_input("To deploy as a service"):
            indicator.update(dict(
                service_type=legal_input(
                    "Choose a service type (cli/flask/nginx) < ",
                    __legal_input={"cli", "flask", "nginx"},
                )
            ))

    if skip_top:
        tar_dir = "./"
    else:
        tar_dir = main_params["project"]
        os.makedirs(tar_dir)

    __project_type = main_params["project_type"]

    project_types[__project_type](tar_dir=tar_dir, main_params=main_params, docs_params=docs_params, **indicator)
