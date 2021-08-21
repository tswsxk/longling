# coding: utf-8
# create by tongshiwei on 2019/4/11

"""
自定义的配置文件及对应的解析工具包。目的是为了更方便、快速地进行文件参数配置与解析。
"""
import argparse
import inspect
import json
import logging
import os
import re
import sys
from copy import deepcopy
from typing import List, Dict, Iterable

import toml
import yaml
from longling import wf_open
from longling.lib.path import path_append

__all__ = [
    "CLASS_EXCLUDE_NAMES", "get_class_var",
    "get_parsable_var", "load_configuration",
    "var2exp", "path_append",
    "Configuration", "ConfigurationParser",
    "Formatter", "ParserGroup", "is_classmethod"
]

CLASS_EXCLUDE_NAMES = set(vars(object).keys()) | {
    '__module__', '__main__', '__dict__', '__weakref__',
}


def get_class_var(class_obj, exclude_names: (set, None) = None, get_vars=None) -> dict:
    """
    Update in v1.3.18

    获取某个类的所有属性的变量名及其值

    Examples
    --------
    >>> class A(object):
    ...     att1 = 1
    ...     att2 = 2
    >>> get_class_var(A)
    {'att1': 1, 'att2': 2}
    >>> get_class_var(A, exclude_names={"att1"})
    {'att2': 2}
    >>> class B(object):
    ...     att3 = 3
    ...     att4 = 4
    ...     @staticmethod
    ...     def excluded_names():
    ...         return {"att4"}
    >>> get_class_var(B)
    {'att3': 3}

    Parameters
    ----------
    class_obj:
        类或类实例。需要注意两者的区别。
    exclude_names:
        需要排除在外的变量名。也可以通过在类定义 excluded_names 方法来指定要排除的变量名。
    get_vars

    Returns
    -------
    class_var:
        类内属性变量名及值

    """
    default_excluded_names = CLASS_EXCLUDE_NAMES if exclude_names is None \
        else CLASS_EXCLUDE_NAMES | exclude_names
    excluded_names = default_excluded_names if not hasattr(
        class_obj, "excluded_names"
    ) else class_obj.excluded_names() | default_excluded_names

    variables = {}
    if get_vars is not False:  # True or None
        variables.update({
            k: v for k, v in vars(class_obj).items() if
            not inspect.isroutine(v) and k not in excluded_names
        })
    if get_vars is not True:  # False or None
        for k in dir(class_obj):
            if hasattr(class_obj, k):
                v = getattr(class_obj, k)
                if not inspect.isroutine(v) and k not in excluded_names:
                    variables[k] = v
    return variables


def parse_params(params: dict, parse_functions: (dict, None) = None) -> dict:
    """
    参数后处理，根据parse_functions中定义的转换函数，对params中的变量值进行转换。
    例如，可用于将无法直接dump的变量转换成可以dump的变量。

    Examples
    --------
    >>> params = {"a": 1}
    >>> parse_functions = {"a": lambda x: x + x}
    >>> parse_params(params, parse_functions)
    {'a': 2}
    >>> params = {"a": 1}
    >>> parse_params(params)
    {'a': 1}

    """
    if parse_functions is not None:
        for name, func in parse_functions.items():
            if name in params:
                params[name] = func(params[name])
    return params


def get_parsable_var(class_obj, parse_exclude: set = None, dump_parse_functions=None, get_vars=True):
    """获取所有可以被解析的参数及其值，可以使用dump_parse_functions来对不可dump的值进行转换"""
    params = get_class_var(class_obj, exclude_names=parse_exclude, get_vars=get_vars)
    return parse_params(params, dump_parse_functions)


def load_configuration(fp, file_format="json", load_parse_function=None):
    """
    装载配置文件

    Updated in version 1.3.16

    Parameters
    ----------
    fp
    file_format
    load_parse_function

    """
    if file_format == "json":
        params = json.load(fp)
    elif file_format == "toml":
        params = toml.load(fp)
    elif file_format == "yaml":
        params = yaml.load(fp, Loader=yaml.FullLoader)
    else:
        raise TypeError("Unsupported file format: %s, only `json`, `toml` and `yaml` are supported" % file_format)
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

    exp = """str(r"%s").format(%s)""" % (
        re.sub(r"\$\w+", "{}", var_str), ",".join(
            [env_wrap(env_var) for env_var in env_vars]
        )
    )
    return exp


def is_classmethod(method):
    """

    Parameters
    ----------
    method

    Returns
    -------

    Examples
    --------
    >>> class A:
    ...     def a(self):
    ...         pass
    ...     @staticmethod
    ...     def b():
    ...         pass
    ...     @classmethod
    ...     def c(cls):
    ...         pass
    >>> obj = A()
    >>> is_classmethod(obj.a)
    False
    >>> is_classmethod(obj.b)
    False
    >>> is_classmethod(obj.c)
    True
    >>> def fun():
    ...     pass
    >>> is_classmethod(fun)
    False
    """
    bound_to = getattr(method, '__self__', None)
    if not isinstance(bound_to, type):
        # must be bound to a class
        return False
    name = method.__name__
    for cls in bound_to.__mro__:
        descriptor = vars(cls).get(name)
        if descriptor is not None:
            return isinstance(descriptor, classmethod)
    return False  # pragma: no cover


class Configuration(object):
    """
    自定义的配置文件基类


    Examples
    --------
    >>> c = Configuration(a=1, b="example", c=[0,2], d={"a1": 3})
    >>> c.instance_var
    {'a': 1, 'b': 'example', 'c': [0, 2], 'd': {'a1': 3}}
    >>> c.default_file_format()
    'json'
    >>> c.get("a")
    1
    >>> c.get("e") is None
    True
    >>> c.get("e", 0)
    0
    >>> c.update(e=2)
    >>> c["e"]
    2
    """

    def __init__(self, logger=logging, **kwargs):
        self.logger = logger

        params = self.class_var

        params.update(**kwargs)
        for param, value in params.items():
            # all class variables will be contained in instance variables
            setattr(self, "%s" % param, value)

    def deep_update(self, **kwargs):
        for param, value in kwargs.items():
            setattr(self, "%s" % param, value)

    def _update(self, **kwargs):
        self.deep_update(**kwargs)

    def update(self, **kwargs):
        params = self.instance_var
        params.update(**kwargs)
        self._update(**params)

    @classmethod
    def vars(cls):
        # todo: check why when deepcopy is missed, two instance will have same id for class_var
        return deepcopy(get_class_var(cls))

    @property
    def class_var(self):
        """
        获取所有设定的参数

        Returns
        -------
        parameters: dict
            all variables used as parameters
        """
        return self.vars()

    @classmethod
    def pvars(cls):
        return get_parsable_var(
            cls,
            parse_exclude=None,
            dump_parse_functions=None,
        )

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

    @property
    def instance_var(self):
        return get_parsable_var(
            self,
            parse_exclude=None,
            dump_parse_functions=None,
        )

    @classmethod
    def load_parse_function(cls):
        return None

    @classmethod
    def default_file_format(cls):
        return "json"

    @classmethod
    def load_cfg(cls, cfg_path, file_format=None, **kwargs):
        """从配置文件中装载配置参数"""
        file_format = file_format if file_format is not None else cls.default_file_format()

        with open(cfg_path) as f:
            params = load_configuration(
                f, file_format=file_format, load_parse_function=cls.load_parse_function()
            )
        params.update(kwargs)
        return params

    @classmethod
    def load(cls, cfg_path, file_format=None, **kwargs):
        """
        从配置文件中装载配置类

        Updated in version 1.3.16
        """
        file_format = file_format if file_format is not None else cls.default_file_format()
        return cls(**cls.load_cfg(cfg_path, file_format=file_format, **kwargs))

    def dump(self, cfg_path: str, override=True, file_format=None):
        """
        将配置参数写入文件

        Updated in version 1.3.16

        Parameters
        ----------
        cfg_path: str
        override: bool
        file_format: str
        """
        if os.path.isfile(cfg_path) and not override:
            self.logger.warning(
                "file %s existed, dump aborted" % os.path.abspath(cfg_path)
            )
            return
        self.logger.info(
            "writing configuration parameters to %s" % os.path.abspath(cfg_path)
        )
        file_format = file_format if file_format is not None else self.default_file_format()

        with wf_open(cfg_path) as wf:
            if file_format == "json":
                json.dump(self.parsable_var, wf, indent=2)
            elif file_format == "toml":
                toml.dump(self.parsable_var, wf)
            elif file_format == "yaml":
                yaml.dump(self.parsable_var, wf)
            else:
                raise TypeError(
                    "Unsupported file format: %s, only `json`, `toml` and `yaml` are supported" % file_format
                )

    def __str__(self):
        string = []
        for k, v in vars(self).items():
            string.append("%s: %s" % (k, v))
        return "\n".join(string)

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        return getattr(self, item)

    def __contains__(self, item):
        return item in vars(self)

    def get(self, k, default=None):
        if k in self:
            return self[k]
        else:
            return default

    def items(self):
        return {k: v for k, v in self.parsable_var.items()}

    @classmethod
    def run_time_variables(cls):
        return {'logger'}

    @classmethod
    def excluded_names(cls):
        """
        获取非参变量集

        Returns
        -------
        exclude names set: set
            所有非参变量
        """
        return CLASS_EXCLUDE_NAMES | {
            'class_var', 'parsable_var', 'instance_var', 'items', 'load', 'dump', 'help_info', 'get', 'update',
            'deep_update'
        } | cls.run_time_variables()

    @classmethod
    def help_info(cls) -> dict:
        return {}


# class InheritableConfiguration(Configuration):
#     """
#     Examples
#     -------
#     >>> class Parent(Configuration):
#     ...     a = 1
#     ...     b = 2
#     >>> class Child1(Parent):
#     ...     a = 0
#     ...     c = 3
#     # >>> c1 = Child1()
#     # >>> c1.parsable_var
#     >>> class Child2(Parent, InheritableConfiguration):
#     ...     c = 3
#     >>> c2 = Child2()
#     >>> c2.parsable_var
#
#     """
#     @classmethod
#     def vars(cls):
#         return get_class_var(cls, get_vars=False)
#
#     @property
#     def parsable_var(self):
#         """
#         获取可以进行命令行设定的参数
#
#         Returns
#         -------
#         store_vars: dict
#             可以进行命令行设定的参数
#         """
#         return get_parsable_var(
#             self,
#             parse_exclude=None,
#             dump_parse_functions=None,
#             get_vars=False
#         )


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
    if re.findall(r"(int|float|dict|list|set|tuple|bool)\(.*\)|None", value):
        value = eval(value)
    return value


def parse_dict_string(string):
    """
    解析字典字符串

    Examples
    --------
    >>> parse_dict_string("a=1;b=int(1);c=list([1]);d=None")
    {'a': '1', 'b': 1, 'c': [1], 'd': None}
    >>> parse_dict_string("")
    """
    if not string:
        return None
    else:
        name_value_items = [
            item.strip().split("=", 1) for item in string.strip().rstrip(";").split(";")
        ]
        return {
            name_value[0]: value_parse(name_value[1])
            for name_value in name_value_items
        }


def args_zips(args=None, defaults=None):
    """组合参数及默认值"""
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
    """
    Update in v1.3.18

    配置文件解析类，可用于构建cli工具。该类首先读入所有目标配置文件类class_obj的所有类属性，解析后生成命令行。
    普通属性参数使用 "--att_name att_value" 来读入。另外提供一个额外参数标记 ‘--kwargs’ 来读入可选参数。
    可选参数格式为 ::

        --kwargs key1=value1;key2=value2;...

    首先生成一个解析类 ::

        cli_parser = ConfigurationParser(Configuration)

    除了解析已有配置文件外，解析类还可以进一步添加函数来生成子解析器 ::

        cli_parser = ConfigurationParser($function)

    或者 ::

        cli_parser = ConfigurationParser([$function1, $function2])

    用以下三种解析方式中任意一种来解析参数：

    * 命令行模式 ::

        cli_parser()

    * 字符串传参模式 ::

        cli_parser('$parameter1 $parameters ...')

    * 列表传参模式 ::

        cli_parser(["--a", "int(1)", "--b", "int(2)"])

    Notes
    -----
    包含以下关键字的字符串会在解析过程中进行类型转换

    int, float, dict, list, set, tuple, None

    Parameters
    ----------
    class_type:
        类。注意是类，不是类实例。
    excluded_names:
        类中不进行解析的变量名集合
    commands:
        待解析的命令函数

    Examples
    --------
    >>> class TestC(Configuration):
    ...     a = 1
    ...     b = 2
    >>> def test_f1(k=1):
    ...     return k
    >>> def test_f2(h=1):
    ...      return h
    >>> def test_f3(m):
    ...      return m
    >>> parser = ConfigurationParser(TestC)
    >>> parser("--a 1 --b 2")
    {'a': '1', 'b': '2'}
    >>> ConfigurationParser.get_cli_cfg(TestC)
    {'a': 1, 'b': 2}
    >>> parser(["--a", "1", "--b", "int(1)"])
    {'a': '1', 'b': 1}
    >>> parser(["--a", "1", "--b", "int(1)", "--kwargs", "c=int(3);d=None"])
    {'a': '1', 'b': 1, 'c': 3, 'd': None}
    >>> parser.add_command(test_f1, test_f2, test_f3)
    >>> parser(["test_f1"])
    {'a': 1, 'b': 2, 'k': 1, 'subcommand': 'test_f1'}
    >>> parser(["test_f2"])
    {'a': 1, 'b': 2, 'h': 1, 'subcommand': 'test_f2'}
    >>> parser(["test_f3", "3"])
    {'a': 1, 'b': 2, 'm': '3', 'subcommand': 'test_f3'}
    >>> parser = ConfigurationParser(TestC, commands=[test_f1, test_f2])
    >>> parser(["test_f1"])
    {'a': 1, 'b': 2, 'k': 1, 'subcommand': 'test_f1'}
    >>> class TestCC:
    ...     c = {"_c": 1, "_d": 0.1}
    >>> parser = ConfigurationParser(TestCC)
    >>> parser("--c _c=int(3);_d=float(0.3)")
    {'c': {'_c': 3, '_d': 0.3}}
    >>> class TestCls:
    ...     def a(self, a=1):
    ...         return a
    ...     @staticmethod
    ...     def b(b=2):
    ...         return b
    ...     @classmethod
    ...     def c(cls, c=3):
    ...         return c
    >>> parser = ConfigurationParser(TestCls, commands=[TestCls.b, TestCls.c])
    >>> parser("b")
    {'b': 2, 'subcommand': 'b'}
    >>> parser("c")
    {'c': 3, 'subcommand': 'c'}
    """

    def __init__(self, class_type, excluded_names: (set, None) = None,
                 commands=None,
                 *args,
                 params_help=None, commands_help=None, override_help=False,
                 **kwargs):
        excluded_names = {
            'logger'
        } if excluded_names is None else excluded_names

        self.__proto_type = argparse.ArgumentParser(add_help=False)

        params = {k: v for k, v in get_class_var(class_type).items()}
        _params_help = {}
        _params_help.update(class_type.help_info() if hasattr(class_type, 'help_info') else {})
        _params_help.update({} if params_help is None else params_help)
        params_help = _params_help

        def get_help_info(default_help_info, param_name=None):
            if not params_help or param_name not in params_help:
                return default_help_info
            else:
                if override_help:
                    return "%s" % params_help[param_name]
                else:
                    return "%s, %s" % (params_help[param_name], default_help_info)

        for param, value in params.items():
            if param in excluded_names:  # pragma: no cover
                continue
            if isinstance(value, dict):
                format_tips = ", dict variables, " \
                              "use format: <key>=<value>(;<key>=<value>)"
                self.__proto_type.add_argument(
                    '--%s' % param,
                    help=get_help_info('set %s, default is %s%s' % (
                        param, value, format_tips
                    ), param),
                    default=value,
                    type=parse_dict_string,
                )
            else:
                self.__proto_type.add_argument(
                    '--%s' % param,
                    help=get_help_info('set %s, default is %s' % (param, value), param),
                    default=value,
                    type=value_parse,
                )
        self.__proto_type.add_argument(
            '--kwargs', required=False,
            help=r"add extra argument here, "
                 r"use format: <key>=<value>(;<key>=<value>)"
        )
        self.sub_command_parsers = None
        super(ConfigurationParser, self).__init__(parents=[self.__proto_type], *args, **kwargs)

        if commands is not None:
            assert isinstance(commands, Iterable)
            self.add_command(*commands, help_info=commands_help)

    def add_command(self, *commands, help_info: (List[Dict], dict, str, None) = None):
        """批量添加子命令解析器"""
        if isinstance(help_info, list):
            for command, _help_info in zip(commands, help_info):
                self._add_subcommand(self.func_spec(command), _help_info)
        else:
            for command in commands:
                self._add_subcommand(self.func_spec(command), help_info)

    @staticmethod
    def parse(arguments):
        """参数后解析"""
        arguments = vars(arguments) if not isinstance(arguments, dict) else arguments
        args_dict = dict()
        for k, v in arguments.items():
            if k == "kwargs" and v:
                kwargs = parse_dict_string(v)
                if kwargs:
                    args_dict.update(kwargs)
            elif k == "kwargs":
                pass
            else:
                args_dict[k] = v
        return args_dict

    @classmethod
    def get_cli_cfg(cls, params_class) -> dict:
        """获取默认配置参数"""
        params_parser = cls(params_class)
        kwargs = params_parser.parse_args([])
        kwargs = params_parser.parse(kwargs)
        return kwargs

    def __call__(self, args: (str, list, None) = None) -> dict:
        if isinstance(args, str):
            args = args.split(" ")
        kwargs = self.parse_args(args)
        kwargs = self.parse(kwargs)
        return kwargs

    def _add_subcommand(self, command_parameters, help_info=None):
        """添加子命令解析器"""

        def get_help_info(default_help_info, command=None):
            if isinstance(help_info, dict):
                return help_info.get(command, default_help_info)
            elif command is None:  # pragma: no cover
                return default_help_info
            else:
                return command

        if self.sub_command_parsers is None:
            self.sub_command_parsers = self.add_subparsers(
                parser_class=argparse.ArgumentParser,
                help='help for sub-command'
            )
        subparsers = self.sub_command_parsers
        subcommand, parameters = command_parameters
        subparser = subparsers.add_parser(
            subcommand, help=get_help_info("%s help" % subcommand, subcommand), parents=[self.__proto_type],
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
        """获取函数参数表"""
        argspec = inspect.getfullargspec(f)
        argspec_args = argspec.args[1:] if is_classmethod(f) else argspec.args

        return f.__name__, args_zips(
            argspec_args, argspec.defaults
        ) + args_zips(
            argspec.kwonlyargs,
            argspec.kwonlydefaults.values() if argspec.kwonlydefaults else []
        )


class ParserGroup(argparse.ArgumentParser):
    """
    >>> class TestC(Configuration):
    ...     a = 1
    ...     b = 2
    >>> def test_f1(k=1):
    ...     return k
    >>> def test_f2(h=1):
    ...      return h
    >>> class TestC2(Configuration):
    ...     c = 3
    >>> parser1 = ConfigurationParser(TestC, commands=[test_f1])
    >>> parser2 = ConfigurationParser(TestC, commands=[test_f2])
    >>> pg = ParserGroup({"model1": parser1, "model2": parser2})
    >>> pg(["model1", "test_f1"])
    {'a': 1, 'b': 2, 'k': 1, 'subcommand': 'test_f1'}
    >>> pg("model2 test_f2")
    {'a': 1, 'b': 2, 'h': 1, 'subcommand': 'test_f2'}
    """

    def __init__(self, parsers: dict, prog=None,
                 usage=None,
                 description=None,
                 epilog=None, add_help=True):
        super(ParserGroup, self).__init__(
            prog=prog,
            usage=usage,
            description=description,
            epilog=epilog,
            add_help=add_help
        )
        self.parsers = parsers

    def parse_args(self, args=None):
        args = sys.argv[1:] if args is None else args
        if args[0] in {"-h", "--help"}:  # pragma: no cover
            return super(ParserGroup, self).parse_args(args)
        else:
            target_parser = self.parsers[args[0]]
            real_args = args[1:]
            return target_parser.parse_args(real_args)

    def __call__(self, args: (str, list, None) = None) -> (dict, None):
        if isinstance(args, str):
            args = args.split(" ")
        args = sys.argv[1:] if args is None else args
        if args[0] in {"-h", "--help"}:  # pragma: no cover
            self.parse_args(args)
            return
        kwargs = self.parse_args(args)
        target_parser = args[0]
        return self.parsers[target_parser].parse(kwargs)

    def format_help(self) -> str:
        # hacking the original method
        formatter = self._get_formatter()

        # usage
        formatter.add_usage(self.usage, self._actions,
                            self._mutually_exclusive_groups)

        # description
        formatter.add_text(self.description)

        # positionals, optionals and user-defined groups
        for title, parser in self.parsers.items():
            formatter.start_section(title)
            formatter.add_text(parser.description)
            for action_group in parser._action_groups:
                formatter.start_section(action_group.title)
                formatter.add_text(action_group.description)
                formatter.add_arguments(action_group._group_actions)
                formatter.end_section()
            formatter.end_section()

        # epilog
        formatter.add_text(self.epilog)

        # determine help from format above
        return formatter.format_help()


class Formatter(object):
    """
    以特定格式格式化字符串

    Examples
    --------
    >>> formatter = Formatter()
    >>> formatter("hello world")
    'hello world'
    >>> formatter = Formatter("hello {}")
    >>> formatter("world")
    'hello world'
    >>> formatter = Formatter("hello {} v{:.2f}")
    >>> formatter("world", 0.2)
    'hello world v0.20'
    >>> formatter = Formatter("hello {1} v{0:.2f}")
    >>> formatter(0.2, "world")
    'hello world v0.20'
    >>> Formatter.format(0.2, "world", formatter="hello {1} v{0:.3f}")
    'hello world v0.200'
    """

    def __init__(self, formatter: (str, None) = None):
        self.formatter = formatter

    def __call__(self, *format_string):
        if self.formatter is None:
            assert len(format_string) == 1, \
                "formatter is None, the input should be a single value, now is %s, " \
                "which has %s value" % (format_string, len(format_string))
            return format_string[0]
        else:
            return self.formatter.format(*format_string)

    @staticmethod
    def format(*format_string, formatter=None):
        formatter = Formatter(formatter)
        return formatter(*format_string)
