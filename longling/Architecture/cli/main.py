# coding: utf-8
# 2019/9/20 @ tongshiwei

__all__ = ["cli"]

from longling.lib.stream import build_dir
from longling.Architecture.cli.utils import legal_input
from longling.Architecture.install_proj import project_types
from longling.Architecture.cli.units import *


def cli(skip_top=True, project=None, **kwargs):
    main_params = main_cli(project, **kwargs)

    kwargs.update(main_params)
    if main_params["project_type"] == "docs" or legal_input("Install docs (y/n, default is y) < ",
                                                            __legal_input={'y', 'n'}, __default_value='y') == 'y':
        docs_params = docs_cli(**kwargs)
    else:
        docs_params = {}

    if skip_top:
        tar_dir = "./"
    else:
        tar_dir = build_dir("%s" % main_params["project"])

    __project_type = main_params["project_type"]

    project_types[__project_type](tar_dir=tar_dir, main_params=main_params, docs_params=docs_params)


if __name__ == '__main__':
    cli()
