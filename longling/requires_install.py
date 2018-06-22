import os

from longling.lib.utilog import config_logging, LogLevel
from longling.base import string_types

logger = config_logging(logger="requires_install", console_log_level=LogLevel.INFO)

REQUIRES_FILE = "requires.txt"


def __find_file(root, name):
    files_path = set()
    for relpath, dirs, files in os.walk(root):
        if name in files:
            full_path = os.path.join(root, relpath, name)
            files_path.add(os.path.normpath(os.path.abspath(full_path)))
    return files_path


def __sperator_replace(input_str: string_types):
    if '.' in input_str:
        input_str.replace(".", os.path.sep)
    return input_str


def run(module_names=None, default_confirm=False):
    assert module_names is None or type(module_names) is list or type(module_names) in string_types
    if type(module_names) in string_types:
        module_names = [module_names]
    logger.info("Confirm Mode: %s, these module requires.txt will be check: %s" % (
        "all confirm without any more permission" if default_confirm
        else "each requires file will need confirmation before installed",
        module_names))
    files_path = set()
    package_set = set()
    for module_name in module_names:
        new_files = __find_file(__sperator_replace(module_name), REQUIRES_FILE)
        for new_file in new_files:
            if new_file not in files_path:
                if default_confirm or input("install all package in %s?(Y/N)" % new_file) in ('Y', 'y', ''):
                    files_path.add(new_file)
    for file_name in files_path:
        with open(file_name) as re_file:
            for line in re_file:
                if line.strip():
                    package_set.add(line.strip().replace("==", " "))
    logger.info("these package will be installed:\n\t%s" % ("\n\t".join(package_set)))
    if default_confirm or input("installed? (Y/N)") in ('Y', 'y', ''):
        logger.info("installing")
        for file_name in files_path:
            os.system("pip install -r %s" % file_name)



