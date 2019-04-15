# coding: utf-8
# create by tongshiwei on 2018/8/5

from __future__ import absolute_import
from __future__ import print_function

import argparse
import os

import longling
from longling.lib.parser import path_append
from longling.lib.stream import wf_open
from longling.lib.utilog import config_logging, LogLevel

__all__ = ["new_model", "cli"]

GLUM_DIR = path_append(
    os.path.dirname(longling.__file__),
    "ML.MxnetHelper.glue.ModelName".replace(".", os.sep),
    to_str=True
)

logger = config_logging(
    logger="glue", level=LogLevel.DEBUG, console_log_level=LogLevel.DEBUG
)


def new_model(model_name, directory=None, skip_top=True):
    target_dir = os.path.join(
        directory, model_name
    ) if directory is not None else model_name

    target_dir = os.path.abspath(target_dir)
    glue_dir = GLUM_DIR if skip_top is False else path_append(
        GLUM_DIR, "ModelName", to_str=True
    )
    if not os.path.exists(glue_dir):
        logger.error(
            "template files does not exist, process aborted, "
            "check the path %s" % glue_dir
        )
        exit(-1)

    logger.info(glue_dir + " -> " + target_dir)
    if os.path.isdir(target_dir):
        logger.error(
            "directory already existed, will not override, generation abort"
        )
        return False
    logger.info("generating file, root path is %s", target_dir)
    big_module_name = "%sModule" % (model_name[0].upper() + model_name[1:])

    def name_replace(name):
        return name.replace(
            "ModelName", model_name
        ).replace("GluonModule", big_module_name)

    for root, dirs, files in os.walk(glue_dir):
        if 'data' + os.sep in root or '_build' in root or os.path.join(
                'docs',
                'source'
        ) in root:
            logger.debug("skip %s-%s" % (root, files))
            continue
        for filename in files:
            if '.pyc' in filename:
                logger.debug("skip %s-%s" % (root, filename))
                continue
            dirname = os.path.abspath(target_dir + os.sep + name_replace(
                root.replace(glue_dir, "")))
            source_file = os.path.abspath(os.path.join(root, filename))
            target_file = os.path.abspath(
                os.path.join(dirname, name_replace(filename)))
            logger.debug(source_file + ' -> ' + target_file)
            with open(source_file, encoding="utf-8") as f, wf_open(
                    target_file) as wf:
                try:
                    for line in f:
                        print(name_replace(line), end="", file=wf)
                except UnicodeDecodeError:
                    print(source_file, line)
                    raise UnicodeDecodeError
    return True


def cli(model_name="longling"):
    parser = argparse.ArgumentParser()
    if model_name:
        parser.add_argument(
            "--model_name", default="%s" % model_name,
            help="set the model name, default is %s" % model_name
        )
    else:
        parser.add_argument(
            "--model_name",
            help="set the model name, default is %s" % model_name,
            required=True
        )

    parser.add_argument(
        "--directory", default=None,
        help="set the directory, default is None"
    )
    parser.add_argument(
        "--skip_top",
        help="whether to skip the top files, like docs",
        action='store_true',
    )

    args = parser.parse_args()

    if new_model(
            model_name=args.model_name, directory=args.directory,
            skip_top=args.skip_top
    ):
        logger.info("success")
    else:
        logger.error("error")


if __name__ == '__main__':
    cli()
