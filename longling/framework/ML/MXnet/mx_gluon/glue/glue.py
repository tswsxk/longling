# coding: utf-8
# create by tongshiwei on 2018/8/5

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
from longling.lib.stream import wf_open
from longling.lib.utilog import config_logging, LogLevel

logger = config_logging(logger="glue", console_log_level=LogLevel.INFO)


def new_module(module_name, directory=None):
    glum_directory = os.path.join(os.path.dirname(sys._getframe().f_code.co_filename), "module_name")
    target_dir = os.path.join(directory, module_name) if directory else module_name
    target_dir = os.path.abspath(target_dir)
    logger.debug(glum_directory, "->", target_dir)
    if os.path.isdir(target_dir):
        logger.error("directory already existed, will not override, generation abort")
        return False
    logger.info("generating file, root path is %s", target_dir)
    big_module_name = "%sModule" % (module_name[0].upper() + module_name[1:])

    def name_replace(name):
        return name.replace("module_name", module_name).replace("GluonModule", big_module_name)

    for root, dirs, files in os.walk(glum_directory):
        if 'data' + os.sep in root or '_build' in root or os.path.join('docs', 'source') in root:
            logger.debug("skip %s-%s" % (root, files))
            continue
        for filename in files:
            if '.pyc' in filename:
                logger.debug("skip %s-%s" % (root, filename))
                continue
            dirname = os.path.abspath(name_replace(root))
            source_file = os.path.abspath(os.path.join(root, filename))
            target_file = os.path.abspath(os.path.join(dirname, name_replace(filename)))
            logger.debug(source_file, '->', target_file)
            with open(source_file, encoding="utf-8") as f, wf_open(target_file) as wf:
                try:
                    for line in f:
                        print(name_replace(line), end="", file=wf)
                except UnicodeDecodeError:
                    print(source_file, line)
                    exit(-1)
                    return False
    return True


if __name__ == '__main__':
    import argparse

    module_name = "SNN"
    parser = argparse.ArgumentParser()
    parser.add_argument("--module_name", default="%s" % module_name,
                        help="set the module name, default is %s" % module_name)

    parser.add_argument("--directory", default=None, help="set the directory, default is None")

    args = parser.parse_args()

    new_module(module_name=args.module_name, directory=args.directory)
