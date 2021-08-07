# coding: utf-8
# create by tongshiwei on 2019-9-4

from __future__ import absolute_import
from __future__ import print_function

import argparse
import os

from longling.lib.parser import path_append
from longling.lib.stream import wf_open
from longling.lib.utilog import config_logging, LogLevel

logger = config_logging(
    logger="glue", level=LogLevel.DEBUG, console_log_level=LogLevel.DEBUG
)


def new_model(model_name, source_dir, directory=None, level="project", skip_existing=False):  # pragma: no cover
    target_dir = os.path.join(
        directory, model_name
    ) if directory is not None else model_name

    target_dir = os.path.abspath(target_dir)
    if level == "project":
        pass
    elif level == "model":
        source_dir = path_append(
            source_dir, "ModelName", to_str=True,
        )
    elif level == "module":
        source_dir = path_append(
            source_dir, "ModelName", "Module", to_str=True,
        )
    else:
        raise ValueError("unknown level: %s" % level)
    if not os.path.exists(source_dir):
        logger.error(
            "template files does not exist, process aborted, "
            "check the path %s" % source_dir
        )
        exit(-1)

    logger.info(source_dir + " -> " + target_dir)
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

    for root, dirs, files in os.walk(source_dir):
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
                root.replace(source_dir, "")))
            source_file = os.path.abspath(os.path.join(root, filename))
            target_file = os.path.abspath(
                os.path.join(dirname, name_replace(filename)))
            logger.debug(source_file + ' -> ' + target_file)
            if os.path.exists(target_file) and skip_existing:
                pass
            else:
                with open(source_file, encoding="utf-8") as f, wf_open(
                        target_file) as wf:
                    try:
                        for line in f:
                            print(name_replace(line), end="", file=wf)
                    except UnicodeDecodeError:
                        print(source_file, line)
                        raise UnicodeDecodeError
    return True


def cli(source_dir, model_name=None):  # pragma: no cover
    parser = argparse.ArgumentParser()
    if model_name is not None:
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
        "--level", choices={"project", "model", "module"},
        help="the level",
        default="model",
    )
    parser.add_argument(
        "--no-skip-existing",
        help="override the existing files",
        action="store_false"
    )

    args = parser.parse_args()

    if new_model(
            model_name=args.model_name, source_dir=source_dir,
            directory=args.directory, level=args.level, skip_existing=not args.no_skip_existing,
    ):
        logger.info("success")
    else:
        logger.error("error")
