# coding: utf-8
# create by tongshiwei on 2019/4/11

import argparse
import inspect
import json
import logging
import os
import re

from longling import wf_open
from longling.lib.path import path_append

__all__ = [
    "CLASS_EXCLUDE_NAMES", "get_class_var",
    "get_parsable_var", "load_configuration_json",
    "var2exp", "path_append",
    "Configuration", "ConfigurationParser"
]

CLASS_EXCLUDE_NAMES = set(vars(object).keys()) | {
    '__module__', '__main__', '__dict__', '__weakref__',
}


def get_class_var(class_type, exclude_names=None):
    default_excluded_names = CLASS_EXCLUDE_NAMES if exclude_names is None \
        else exclude_names
    excluded_names = default_excluded_names if not hasattr(
        class_type, "excluded_names"
    ) else class_type.excluded_names() | default_excluded_names

    variables = {
        k: v for k, v in vars(class_type).items() if
        not inspect.isroutine(v) and k not in excluded_names
    }
    return variables


def parse_params(params, parse_functions=None):
    if parse_functions is not None:
        for name, func in parse_functions.items():
            if name in params:
                params[name] = func(params[name])
    return params


def get_parsable_var(class_obj, parse_exclude=None, dump_parse_functions=None):
    params = get_class_var(class_obj, exclude_names=parse_exclude)
    return parse_params(params, dump_parse_functions)


def load_configuration_json(fp, load_parse_function=None):
    params = json.load(fp)
    return parse_params(params, load_parse_function)


def var2exp(var_str, env_wrap=lambda x: x):
    """
    将含有 $ 标识的变量转换为表达式

    Parameters
    ----------
    var_str
    env_wrap

    Examples
    --------
    >>> root = "dir"
    >>> dataset = "d1"
    >>> eval(var2exp("$root/data/$dataset"))
    'dir/data/d1'
    """
    var_str = str(var_str)

    pattern = re.compile(r"\$(\w+)")

    env_vars = pattern.findall(var_str)

    exp = """str("%s").format(%s)""" % (
        re.sub(r"\$\w+", "{}", var_str), ",".join(
            [env_wrap(env_var) for env_var in env_vars]
        )
    )
    return exp


class Configuration(object):
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
            parse_exclude=None,
            dump_parse_functions=None,
        )

    @staticmethod
    def load_cfg(params_json, **kwargs):
        with open(params_json) as f:
            params = load_configuration_json(
                f, load_parse_function=None
            )
        params.update(kwargs)
        return params

    @staticmethod
    def load(cfg_path, **kwargs):
        Configuration(Configuration.load_cfg(cfg_path, **kwargs))

    def dump(self, cfg_json, override=False):
        if os.path.isfile(cfg_json) and not override:
            self.logger.warning("file %s existed, dump aborted" % cfg_json)
            return
        self.logger.info("writing configuration parameters to %s" % cfg_json)
        with wf_open(cfg_json) as wf:
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


def value_parse(value):
    r"""
    将含有关键字的字符串转换为关键字指定的类型
    支持的关键字类型为python的基础数据类型:
    int, float, dict, list, set, tuple, None

    Parameters
    ----------
    value: str
        字符串值
    Examples
    --------
    >>> value_parse("int(1.0)")
    1
    >>> value_parse("float(1.0)")
    1.0
    >>> value_parse("dict(a=1, b=2.0, c='d')")
    {'a': 1, 'b': 2.0, 'c': 'd'}
    >>> value_parse("list([1, 2, 3, 4])")
    [1, 2, 3, 4]
    >>> value_parse("set([1, 1, 2, 3])")
    {1, 2, 3}
    >>> value_parse("tuple([1, 2, 3])")
    (1, 2, 3)
    >>> value_parse("None") is None
    True
    """
    if re.findall(r"(int|float|dict|list|set|tuple)\(.*\)|None", value):
        value = eval(value)
    return value


def parse_dict_string(string):
    if not string:
        return None
    else:
        name_value_items = [
            item.strip().split("=") for item in string.strip().split(",")
        ]
        return {
            name_value[0]: value_parse(name_value[1])
            for name_value in name_value_items
        }


def args_zips(args=None, defaults=None):
    args = [] if args is None else args
    defaults = [] if defaults is None else defaults
    assert len(args) >= len(defaults)
    zipped_args = []
    zipped_args.extend([
        {"name": arg} for arg in args[:len(args) - len(defaults)]
    ])
    zipped_args.extend([
        {"name": arg, "default": default} for arg, default in zip(
            args[len(args) - len(defaults):], defaults
        )
    ])
    return zipped_args


class ConfigurationParser(argparse.ArgumentParser):
    def __init__(self, class_obj, excluded_names=None, *args, **kwargs):
        excluded_names = {
            'logger'
        } if excluded_names is None else excluded_names

        super(ConfigurationParser, self).__init__(*args, **kwargs)
        params = {k: v for k, v in get_class_var(class_obj).items()}
        for param, value in params.items():
            if param in excluded_names:
                continue
            if isinstance(value, dict):
                format_tips = ", dict variables, " \
                              "use format: <key>=<value>(,<key>=<value>)"
                self.add_argument(
                    '--%s' % param,
                    help='set %s, default is %s%s' % (
                        param, value, format_tips
                    ), default=value,
                    type=parse_dict_string,
                )
            else:
                self.add_argument(
                    '--%s' % param,
                    help='set %s, default is %s' % (param, value),
                    default=value,
                    type=value_parse,
                )
        self.add_argument(
            '--kwargs', required=False,
            help=r"add extra argument here, "
                 r"use format: <key>=<value>(,<key>=<value>)"
        )
        self.sub_command_parsers = None

    @staticmethod
    def parse(arguments):
        arguments = vars(arguments)
        args_dict = dict()
        for k, v in arguments.items():
            if k == "kwargs" and v:
                kwargs = parse_dict_string(v)
                if kwargs:
                    args_dict.update(kwargs)
                continue
            args_dict[k] = v
        return args_dict

    @staticmethod
    def get_cli_cfg(params_class):
        params_parser = ConfigurationParser(params_class)
        kwargs = params_parser.parse_args()
        kwargs = params_parser.parse(kwargs)
        return kwargs

    def __call__(self, args=None):
        kwargs = self.parse_args(args)
        kwargs = self.parse(kwargs)
        return kwargs

    def add_subcommand(self, command_parameters):
        if self.sub_command_parsers is None:
            self.sub_command_parsers = self.add_subparsers(
                parser_class=argparse.ArgumentParser,
                help='help for sub-command'
            )
        subparsers = self.sub_command_parsers
        subcommand, parameters = command_parameters
        subparser = subparsers.add_parser(
            subcommand, help="%s help" % subcommand
        )
        subparser.set_defaults(subcommand=subcommand)
        for parameter in parameters:
            if "default" in parameter:
                subparser.add_argument(
                    "--%s" % parameter["name"],
                    default=parameter["default"],
                    required=False,
                    type=value_parse,
                    help="set %s, default is %s" % (
                        parameter["name"], parameter["default"]
                    )
                )
            else:
                subparser.add_argument(
                    parameter["name"],
                    type=value_parse,
                    help="set %s" % parameter["name"]
                )

    @staticmethod
    def func_spec(f):
        argspec = inspect.getfullargspec(f)
        return f.__name__, args_zips(
            argspec.args, argspec.defaults
        ) + args_zips(
            argspec.kwonlyargs,
            argspec.kwonlydefaults.values() if argspec.kwonlydefaults else []
        )


if __name__ == '__main__':
    import doctest

    doctest.testmod()
