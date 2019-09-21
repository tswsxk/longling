# coding: utf-8
# 2019/9/21 @ tongshiwei

__all__ = ["main_cli", "docs_cli"]

import datetime
from longling.Architecture.install_proj import project_types
from longling.Architecture.cli.utils import legal_input


def main_cli(project, **kwargs):
    params = dict(
        project=legal_input("Project Name < ", is_legal=lambda x: True if x else False) if not project else project,
        project_type=legal_input("Project Type (%s) < " % "/".join(project_types), __legal_input=project_types),
    )
    return params


def docs_cli(project=None, title=None, author=None, copyright=None, **kwargs):
    params = dict(
        project=legal_input("Project Name < ", is_legal=lambda x: True if x else False) if not project else project,
        docs_root="docs/" if legal_input("mkdir docs? (y/n, default is n) < ",
                                         __legal_input={"y", "n"}, __default_value='n') == "y" else "./"
    )
    params.update(dict(
        title=legal_input("Docs Title (default is %s) < " % params["project"],
                          __default_value=params["project"]) if not title else title,
        author=legal_input("Author < ", is_legal=lambda x: True if x else False) if not author else author,

    ))

    default_copyright = "%s, %s" % (datetime.date.today().year, params["author"])
    params.update(dict(
        copyright=legal_input("Copyright (default is %s) < " % default_copyright,
                              __default_value=default_copyright) if not copyright else copyright,
        readthedocs=True if legal_input("Install .readthedocs.yml (y/n, default is y) < ", __legal_input={'y', 'n'},
                                        __default_value='y') == 'y' else False
    ))

    return params
