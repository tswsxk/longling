# coding:utf-8
# created by tongshiwei on 2018/10/15
"""
参数模板
"""

from __future__ import absolute_import
from __future__ import print_function

import argparse
import inspect
import os
import datetime

import yaml

from longling.lib.utilog import config_logging, LogLevel
from longling.lib.stream import wf_open


class Parameters(object):
    # default path
    root = os.path.abspath("./") + os.sep
    model_name = "modelName"
    # Can also specify the model_name using explicit string
    # model_name = "module_name"
    dataset = ""

    if dataset:
        root_data_dir = os.path.abspath(os.path.join(root, "data")) + os.sep
    else:
        root_data_dir = os.path.abspath(os.path.join(root, "data" + os.sep + "{}".format(dataset))) + os.sep

    data_dir = os.path.abspath(os.path.join(root_data_dir, "data")) + os.sep
    model_dir = os.path.abspath(os.path.join(root_data_dir, model_name)) + os.sep

    logger = config_logging(logger=model_name, console_log_level=LogLevel.INFO)
    time_stamp = False

    param = {

    }

    def __init__(self, params_yaml=None, **kwargs):
        params = self.class_var
        if params_yaml:
            params.update(self.load(params_yaml=params_yaml))
        params.update(**kwargs)

        for param, value in params.items():
            setattr(self, "%s" % param, value)

        # rebuild relevant directory or file path according to the kwargs
        if 'root_data_dir' not in params:
            if self.dataset:
                self.root_data_dir = os.path.abspath(os.path.join(self.root, "data")) + os.sep
            else:
                self.root_data_dir = os.path.abspath(
                    os.path.join(self.root, "data" + os.sep + "{}".format(self.dataset))) + os.sep
        if 'data_dir' not in params:
            self.data_dir = os.path.abspath(os.path.join(self.root_data_dir, "data")) + os.sep
        if 'model_dir' not in params:
            if hasattr(self, 'time_stamp') and self.time_stamp and hasattr(self, 'model_dir'):
                time_stamp = "_%s" % datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                self.model_dir = os.path.abspath(
                    os.path.join(self.root_data_dir, self.model_name)) + time_stamp + os.sep
            else:
                self.model_dir = os.path.abspath(os.path.join(self.data_dir, self.model_name)) + os.sep
        self.validation_result_file = os.path.abspath(os.path.join(self.model_dir, "result.json"))

    def items(self):
        return {k: v for k, v in vars(self).items() if k not in {'logger'}}

    @staticmethod
    def load(params_yaml, logger=None):
        f = open(params_yaml)
        params = yaml.load(f)
        f.close()
        if logger:
            params['logger'] = logger
        return params

    def __str__(self):
        return str(self.parsable_var)

    def dump(self, param_yaml, override=False):
        if os.path.isfile(param_yaml) and not override:
            self.logger.warning("file %s existed, dump aborted" % param_yaml)
            return
        self.logger.info("writing parameters to %s" % param_yaml)
        with wf_open(param_yaml) as wf:
            dump_data = yaml.dump(self.parsable_var, default_flow_style=False)
            print(dump_data, file=wf)
            self.logger.info(dump_data)

    @property
    def class_var(self):
        variables = {k: v for k, v in vars(type(self)).items() if
                     not inspect.isroutine(v) and k not in self.excluded_names()}
        return variables

    @property
    def parsable_var(self):
        store_vars = {k: v for k, v in vars(self).items() if k not in {'logger'}}
        return store_vars

    @staticmethod
    def excluded_names():
        """
        获取非参变量集
        Returns
        -------
        exclude names set: set
            所有非参变量
        """
        return {'__doc__', '__module__', '__dict__', '__weakref__',
                'class_var', 'parsable_var'}


class ParameterParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ParameterParser, self).__init__(*args, **kwargs)
        self.add_argument('--root_prefix', dest='root_prefix', default='', help='set root prefix')
        params = {k: v for k, v in vars(Parameters).items() if
                  not inspect.isroutine(v) and k not in Parameters.excluded_names()}
        for param, value in params.items():
            if param == 'logger':
                continue
            self.add_argument('--%s' % param, help='set %s, default is %s' % (param, value), default=value)
        self.add_argument('--kwargs', required=False, help=r"add extra argument here, use format: <key>=<value>")

    @staticmethod
    def parse(arguments):
        arguments = vars(arguments)
        args_dict = dict()
        for k, v in arguments.items():
            if k in {'root_prefix'}:
                continue
            args_dict[k] = v
        if arguments['root_prefix']:
            args_dict['root'] = os.path.abspath(os.path.join(arguments['root_prefix'], arguments['root']))

        return args_dict


if __name__ == '__main__':
    default_yaml_file = os.path.join(Parameters.model_dir, "parameters.yaml")

    # 命令行参数配置
    parser = ParameterParser()
    kwargs = parser.parse_args()
    kwargs = parser.parse(kwargs)

    data_dir = os.path.join(Parameters.root, "data")
    parameters = Parameters(
        **kwargs
    )
    parameters.dump(default_yaml_file, override=True)
    try:
        logger = parameters.logger
        parameters.load(default_yaml_file, logger=logger)
        parameters.logger.info('format check done')
    except Exception as e:
        print("parameters format error, may contain illegal data type")
        raise e
