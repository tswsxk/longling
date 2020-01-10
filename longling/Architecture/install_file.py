# coding: utf-8
# 2019/9/20 @ tongshiwei

__all__ = [
    "gitignore",
    "pytest", "coverage",
    "pysetup", "sphinx_conf", "makefile",
    "readthedocs", "travis", "nni",
    "template_copy",
]

import functools
from shutil import copyfile

from longling import wf_open, PATH_TYPE
from longling.lib.path import abs_current_dir, path_append
from longling.lib.process_pattern import default_variable_replace as dvr
from longling.lib.utilog import config_logging, LogLevel

logger = config_logging(logger="arch", console_log_level=LogLevel.INFO)

META = path_append(abs_current_dir(__file__), "meta_docs")
default_variable_replace = functools.partial(dvr, quotation="\'")


def template_copy(src: PATH_TYPE, tar: PATH_TYPE, default_value: (str, dict, None) = "", quotation="\'", **variables):
    """
    Generate the tar file based on the template file where the variables will be replaced.
    Usually, the variable is specified like `$PROJECT` in the template file.


    Parameters
    ----------
    src: template file
    tar: target location
    default_value: the default value
    quotation: the quotation to wrap the variable value
    variables: the real variable values which are used to replace the variable in template file
    """
    with open(src) as f, wf_open(tar) as wf:
        for line in f:
            print(
                default_variable_replace(line, default_value=default_value, quotation=quotation, **variables),
                end='', file=wf
            )


def gitignore(atype: str = "", tar_dir: PATH_TYPE = "./"):
    """

    Parameters
    ----------
    atype: the gitignore type, currently support `docs` and `python`
    tar_dir: target directory

    """
    src = path_append(META, "gitignore", "%s.gitignore" % atype)
    tar = path_append(tar_dir, ".gitignore")
    logger.info("gitignore: copy %s -> %s" % (src, tar))
    copyfile(src, tar)


def pytest(tar_dir: PATH_TYPE = "./"):
    src = path_append(META, "pytest.ini")
    tar = path_append(tar_dir, "pytest.ini")
    logger.info("pytest: copy %s -> %s" % (src, tar))
    copyfile(src, tar)


def coverage(tar_dir: PATH_TYPE = "./", **variables):
    src = path_append(META, "setup.cfg.template")
    tar = path_append(tar_dir, "setup.cfg")
    logger.info("coverage: template %s -> %s" % (src, tar))
    template_copy(src, tar, quotation="", **variables)


def pysetup(tar_dir="./", **variables):
    src = path_append(META, "setup.py.template")
    tar = path_append(tar_dir, "setup.py")
    logger.info("pysetup: template %s -> %s" % (src, tar))
    template_copy(src, tar, **variables)


def sphinx_conf(tar_dir="./", **variables):
    src = path_append(META, "docs/conf.py.template")
    tar = path_append(tar_dir, "conf.py")
    logger.info("sphinx_conf: template %s -> %s" % (src, tar))
    template_copy(src, tar, **variables)

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


def travis(tar_dir: PATH_TYPE = "./"):
    src = path_append(META, ".travis.yml")
    tar = path_append(tar_dir, ".travis.yml")
    logger.info("travis: copy %s -> %s" % (src, tar))
    copyfile(src, tar)


def gitlab_ci():
    pass


def makefile(tar_dir="./", **variables):
    src = path_append(META, "Makefile.template")
    tar = path_append(tar_dir, "Makefile")
    logger.info("makefile: template %s -> %s" % (src, tar))
    template_copy(src, tar, default_value=None, quotation='', **variables)


def nni(tar_dir="./"):
    src_dir = path_append(META, "nni")
    for file in ["config.yml", "search_space.json"]:
        copyfile(path_append(src_dir, file), path_append(tar_dir, file))
