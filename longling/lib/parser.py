# coding: utf-8
# create by tongshiwei on 2019/4/11

__all__ = [
    "CLASS_EXCLUDE_NAMES", "get_class_var",
    "get_parsable_var", "load_parameters_json",
    "var2exp", "path_append",
    "Parameters", "ParameterParser"
]

import argparse
import inspect
import json
import logging
import os
import re
from pathlib import PurePath

from longling import wf_open

CLASS_EXCLUDE_NAMES = {'__doc__', '__module__', '__dict__', '__weakref__'}


def get_class_var(class_obj, exclude_names=None):
    default_excluded_names = CLASS_EXCLUDE_NAMES if exclude_names is None \
        else exclude_names
    excluded_names = default_excluded_names if not hasattr(
        class_obj, "excluded_names"
    ) else class_obj.excluded_names()

    variables = {
        k: v for k, v in vars(type(class_obj)).items() if
        not inspect.isroutine(v) and k not in excluded_names
    }
    return variables


def parse_params(params, parse_functions=None):
    if parse_functions is not None:
        for name, func in parse_functions.items():
            if name in params:
                params[name] = func(params[name])
    return params


def get_parsable_var(class_obj, exclude_names=None, dump_parse_functions=None):
    params = get_class_var(class_obj, exclude_names)
    return parse_params(params, dump_parse_functions)


def load_parameters_json(fp, load_parse_function=None):
    params = json.load(fp)
    return parse_params(params, load_parse_function)


def var2exp(var_str):
    var_str = str(var_str)

    pattern = re.compile("\$(\w+)")

    env_vars = pattern.findall(var_str)

    exp = """str("%s").format(%s)""" % (
        re.sub("\$\w+", "{}", var_str), ",".join(env_vars)
    )
    return exp


def path_append(path, addition=None, to_str=False):
    """

    Parameters
    ----------
    path: str or PurePath
    addition: str or PurePath
    to_str: bool
        Convert the new path to str
    Returns
    -------

    """
    path = PurePath(path)
    new_path = path / addition if addition else path
    if to_str:
        return str(new_path)
    return new_path


class Parameters(object):
    def __init__(self, logger=logging):
        self.logger = logger

    @property
    def class_var(self):
        """
        获取所有设定的参数

        Returns
        -------
        parameters: dict
            all variables used as parameters
        """
        return get_class_var(self)

    @property
    def parsable_var(self):
        """
        获取可以进行命令行设定的参数

        Returns
        -------
        store_vars: dict
            可以进行命令行设定的参数
        """
        return get_parsable_var(
            self,
            exclude_names=None,
            dump_parse_functions=None,
        )

    @staticmethod
    def load(params_json):
        with open(params_json) as f:
            params = load_parameters_json(
                f, load_parse_function=None
            )

        return params

    def dump(self, param_json, override=False):
        if os.path.isfile(param_json) and not override:
            self.logger.warning("file %s existed, dump aborted" % param_json)
            return
        self.logger.info("writing parameters to %s" % param_json)
        with wf_open(param_json) as wf:
            json.dump(self.parsable_var, wf, indent=4)

    def __str__(self):
        return str(self.parsable_var)

    def __getitem__(self, item):
        return getattr(self, item)

    def items(self):
        return {k: v for k, v in self.parsable_var}

    @staticmethod
    def excluded_names():
        """
        获取非参变量集

        Returns
        -------
        exclude names set: set
            所有非参变量
        """
        return CLASS_EXCLUDE_NAMES | {
            'class_var', 'parsable_var', 'items', 'load', 'dump'
        }


class ParameterParser(argparse.ArgumentParser):
    def __init__(self, class_obj, excluded_names=None, *args, **kwargs):
        excluded_names = {
            'logger'
        } if excluded_names is None else excluded_names

        super(ParameterParser, self).__init__(*args, **kwargs)
        params = {k: v for k, v in get_class_var(class_obj).items()}
        for param, value in params.items():
            if param in excluded_names:
                continue
            self.add_argument('--%s' % param,
                              help='set %s, default is %s' % (param, value),
                              default=value)
        self.add_argument(
            '--kwargs', required=False,
            help=r"add extra argument here, use format: <key>=<value>"
        )

    @staticmethod
    def parse(arguments):
        arguments = vars(arguments)
        args_dict = dict()
        for k, v in arguments.items():
            args_dict[k] = v

        return args_dict
