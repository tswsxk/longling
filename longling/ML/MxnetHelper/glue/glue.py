# coding: utf-8
# create by tongshiwei on 2018/8/5

from __future__ import absolute_import
from __future__ import print_function

import longling
import os
from longling.lib.stream import wf_open
from longling.lib.utilog import config_logging, LogLevel

glum_directory = os.path.abspath(
    os.path.join(os.path.dirname(longling.__file__),
                 "framework.ML.MXnet.mx_gluon.glue.module_name".replace(".", os.sep)))
logger = config_logging(logger="glue", level=LogLevel.DEBUG, console_log_level=LogLevel.DEBUG)


def new_module(module_name, directory=None):
    target_dir = os.path.join(directory, module_name) if directory else module_name
    target_dir = os.path.abspath(target_dir)
    logger.info(glum_directory + " -> " + target_dir)
    if os.path.isdir(target_dir):
        logger.error("directory already existed, will not override, generation abort")
        return False
    logger.info("generating file, root path is %s", target_dir)
    big_module_name = "%sModule" % (module_name[0].upper() + module_name[1:])

    def name_replace(name):
        return name.replace("module_name", module_name).replace("GluonModule", big_module_name)

    print(glum_directory)
    for root, dirs, files in os.walk(glum_directory):
        if 'data' + os.sep in root or '_build' in root or os.path.join('docs', 'source') in root:
            logger.debug("skip %s-%s" % (root, files))
            continue
        for filename in files:
            if '.pyc' in filename:
                logger.debug("skip %s-%s" % (root, filename))
                continue
            dirname = os.path.abspath(target_dir + os.sep + name_replace(root.replace(glum_directory, "")))
            source_file = os.path.abspath(os.path.join(root, filename))
            target_file = os.path.abspath(os.path.join(dirname, name_replace(filename)))
            logger.debug(source_file + ' -> ' + target_file)
            with open(source_file, encoding="utf-8") as f, wf_open(target_file) as wf:
                try:
                    for line in f:
                        print(name_replace(line), end="", file=wf)
                except UnicodeDecodeError:
                    print(source_file, line)
                    raise UnicodeDecodeError
    return True


if __name__ == '__main__':
    import argparse

    module_name = "longling"
    parser = argparse.ArgumentParser()
    if module_name:
        parser.add_argument("--module_name", default="%s" % module_name,
                            help="set the module name, default is %s" % module_name)
    else:
        parser.add_argument("--module_name", help="set the module name, default is %s" % module_name, required=True)

    parser.add_argument("--directory", default=None, help="set the directory, default is None")

    args = parser.parse_args()

    if new_module(module_name=args.module_name, directory=args.directory):
        logger.info("success")
    else:
        logger.error("error")
