# coding: utf-8
# 2019/9/21 @ tongshiwei

__all__ = ["main_cli", "docs_cli"]

import os
import pathlib
import datetime
from longling.Architecture.install_proj import project_types
from longling.Architecture.cli.utils import legal_input, binary_legal_input


def main_cli(skip_top, project, **kwargs):  # pragma: no cover
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
             **kwargs):  # pragma: no cover
    params = dict(
        project=legal_input("Project Name < ", is_legal=lambda x: True if x else False) if not project else project,
        docs_style=legal_input("Docs Style (mxnet/sphinx, default is %s)? < " % default_style,
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
    params.update(dockerfile=binary_legal_input("Install Dockerfile for documents?"))

    return params
