from __future__ import print_function

import pathlib
import argparse
import os

from longling.base import string_types
from longling.lib.utilog import config_logging, LogLevel

logger = config_logging(
    logger="dependency", console_log_level=LogLevel.INFO
)

REQUIREMENTS_FILE = "requirements"


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module_name", required=False,
        nargs='+',
        default=None,
    )
    args = parser.parse_args()
    find_dependency(args.module_name)


def requirements2list(requirements_file):
    return [
        elem for elem in open(requirements_file).read().split("\n") if elem
    ]


def __find_file(root, name):
    files_path = set()
    for relpath, dirs, files in os.walk(root):
        if name in files:
            full_path = os.path.join(relpath, name)
            files_path.add(os.path.normpath(os.path.abspath(full_path)))
            logger.debug("found %s" % full_path)
    return files_path


def __sperator_replace(input_str: string_types):
    if not input_str:
        return os.path.join(".", os.path.sep)
    if '.' in input_str:
        return input_str.replace(".", os.path.sep)
    return input_str


def find_dependency(module_names=None):
    """

    Parameters
    ----------
    module_names
        root module names, can be string or list,
        when not specified, that is None, use "longling"

    Returns
    -------

    """
    assert module_names is None or type(module_names) is list or type(
        module_names) in string_types
    if module_names is None:
        module_names = "longling"
    if type(module_names) in string_types:
        module_names = [module_names]
    logger.info(
        "These module %s will be check: %s" % (
            REQUIREMENTS_FILE,
            module_names
        )
    )
    root = pathlib.PurePath(__file__).parent
    files_path = set()
    package_set = set()
    for module_name in module_names:
        module_name = module_name.strip("longling.")
        path = root
        if module_name:
            path = root / __sperator_replace(module_name)
        logger.info("searching %s" % path)
        new_files = __find_file(path, REQUIREMENTS_FILE)
        for new_file in new_files:
            if new_file not in files_path:
                files_path.add(new_file)
    logger.info(
        "%s requires files have been found:\n\t%s" % (
            len(files_path), "\n\t".join(files_path))
    )
    for file_name in files_path:
        try:
            for req in requirements2list(file_name):
                package_set.add(req)
        except Exception as e:
            logger.error(e)
    package_set -= {"longling"}
    logger.info(
        "%s dependency packages have be founded:\n\t%s" % (
            len(package_set), ",".join(package_set)
        )
    )

    return package_set


if __name__ == '__main__':
    logger.setLevel(LogLevel.WARN)
    print("\n".join(find_dependency()))
