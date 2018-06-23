from __future__ import print_function

import os

from longling.base import string_types
from longling.lib.stream import wf_open
from longling.lib.utilog import config_logging, LogLevel

logger = config_logging(logger="requires_install", console_log_level=LogLevel.INFO)

REQUIRES_FILE = "requires.txt"


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


def run(module_names=None, default_confirm=True, user_mode=True, source=None, rfile_path="requires.txt"):
    '''
    依赖包检查安装
    :param module_names: root module names, can be string or list, when not specified, that is None, use "longling"
    :param default_confirm: tag to indicating whether or not to confirm every requires.txt installed
    :param user_mode: whether to use "--user" parameter in pip cmd
    :param source: can be http://pypi.mirrors.ustc.edu.cn/simple
    :param rfile_path: when specified, output the result to the file. default "requires.txt"
    :return:
    '''
    assert module_names is None or type(module_names) is list or type(module_names) in string_types
    if module_names is None:
        module_names = "longling"
    if type(module_names) in string_types:
        module_names = [module_names]
    logger.info("require file path is %s%s" % (
        rfile_path, " all depending packages will be installed" if not rfile_path else ""))
    logger.info("Confirm Mode: %s\nthese module requires.txt will be check: %s" % (
        "all confirm without any more permission" if default_confirm
        else "each requires file will need confirmation before installed",
        module_names))
    logger.info("user mode: %s" % user_mode)
    logger.info("specify source: %s" % "Auto" if not source else source)
    files_path = set()
    package_set = set()
    for module_name in module_names:
        module_name = module_name.strip("longling.")
        path = __sperator_replace(module_name)
        logger.info("searching %s" % path)
        new_files = __find_file(path, REQUIRES_FILE)
        for new_file in new_files:
            if new_file not in files_path:
                if default_confirm or input("install all package in %s?(Y/N)\n" % new_file) in ('Y', 'y', ''):
                    files_path.add(new_file)
    logger.info("%s requires files have been found:\n\t%s" % (len(files_path), "\n\t".join(files_path)))
    for file_name in files_path:
        try:
            with open(file_name) as re_file:
                for line in re_file:
                    if line.strip():
                        package_set.add(line.strip())
        except Exception as e:
            logger.error(e)
    logger.info("%s packages will be installed:\n\t%s" % (len(package_set), "\n\t".join(package_set)))
    if not rfile_path and (package_set and (default_confirm or input("installed? (Y/N)\n") in ('Y', 'y', ''))):
        logger.info("installing")
        install_command = "pip install -r {}"
        if user_mode:
            install_command += " --user"
        if source:
            install_command += " -i %s" % source
        for file_name in files_path:
            os.system(install_command.format(file_name))
            # print(install_command.format(file_name))

    if rfile_path and package_set:
        logger.info("output required packages to %s" % rfile_path)
        with wf_open(rfile_path) as wf:
            for package in package_set:
                print(package, file=wf)

if __name__ == '__main__':
    run()
