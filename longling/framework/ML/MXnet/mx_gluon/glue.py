# coding: utf-8
# created by tongshiwei on 18-2-3
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
from longling.lib.stream import wf_open
from longling.lib.utilog import config_logging, LogLevel


logger = config_logging(logger="glue", console_log_level=LogLevel.INFO)


def new_module(module_name, directory=None):
    glum_directory = os.path.dirname(sys._getframe().f_code.co_filename)
    glum_py = os.path.join(glum_directory, "glum.py")
    module_filename = module_name + ".py"
    target = os.path.join(directory, module_filename) if directory else module_filename
    if os.path.isfile(target):
        logger.error("file already existed, will not override, generation abort")
        return False
    logger.info("generating file, path is %s", target)
    big_module_name = "%sModule" % (module_name[0].upper() + module_name[1:])
    with open(glum_py, encoding="utf-8") as f, wf_open(target) as wf:
        for line in f:
            print(line.replace("module_name", module_name).replace("GluonModule", big_module_name), end="", file=wf)
    return True


if __name__ == '__main__':
    new_module("SNN")
