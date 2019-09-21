# coding: utf-8
# 2019/9/20 @ tongshiwei

__all__ = ["gitignore", "pytest", "pysetup", "sphinx_conf", "readthedocs"]

import functools
from shutil import copyfile

from longling import wf_open
from longling.lib.path import abs_current_dir, path_append
from longling.lib.process_pattern import default_variable_replace
from longling.lib.utilog import config_logging, LogLevel

logger = config_logging(logger="arch", console_log_level=LogLevel.INFO)

META = path_append(abs_current_dir(__file__), "meta_docs")
default_variable_replace = functools.partial(default_variable_replace, quotation="\'")


def _template_copy(src, tar, **variables):
    with open(src) as f, wf_open(tar) as wf:
        for line in f:
            print(default_variable_replace(line, default_value="", **variables), end='', file=wf)


def gitignore(atype="", tar_dir="./"):
    src = path_append(META, "gitignore", "%s.gitignore" % atype)
    tar = path_append(tar_dir, ".gitignore")
    logger.info("gitignore: copy %s -> %s" % (src, tar))
    copyfile(src, tar)


def pytest(tar_dir="./"):
    src = path_append(META, "pytest.ini")
    tar = path_append(tar_dir, "pytest.ini")
    logger.info("pytest: copy %s -> %s" % (src, tar))
    copyfile(src, tar)


def coverage():
    pass


def pysetup(tar_dir="./", **variables):
    src = path_append(META, "setup.py.template")
    tar = path_append(tar_dir, "setup.py")
    logger.info("pysetup: template %s -> %s" % (src, tar))
    _template_copy(src, tar, **variables)


def sphinx_conf(tar_dir="./", **variables):
    src = path_append(META, "docs/conf.py.template")
    tar = path_append(tar_dir, "conf.py")
    logger.info("sphinx_conf: template %s -> %s" % (src, tar))
    _template_copy(src, tar, **variables)

    src = path_append(META, "docs/.math.json")
    tar = path_append(tar_dir, ".math.json")
    logger.info("sphinx_conf: copy %s -> %s" % (src, tar))
    copyfile(src, tar)


def readthedocs(tar_dir="./"):
    src = path_append(META, "docs/.readthedocs.yml")
    tar = path_append(tar_dir, ".readthedocs.yml")
    logger.info("readthedocs: copy %s -> %s" % (src, tar))
    copyfile(src, tar)


def docker():
    pass


def travis():
    pass


def gitlab_ci():
    pass
