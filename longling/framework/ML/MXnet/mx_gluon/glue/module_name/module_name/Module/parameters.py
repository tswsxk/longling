# coding: utf-8
# Copyright @tongshiwei

from __future__ import absolute_import
from __future__ import print_function

import argparse
import inspect
import os
import datetime

import yaml

from longling.lib.utilog import config_logging, LogLevel
from longling.lib.stream import wf_open
from longling.framework.ML.MXnet.mx_gluon.glue.parser import MXCtx

from mxnet import cpu, gpu


class Parameters(object):
    # 模型名和数据集设置
    model_name = os.path.basename(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    # Can also specify the model_name using explicit string
    # model_name = "module_name"
    dataset = ""

    # 项目根目录
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")) + os.sep

    # 数据根目录（存储数据集数据及以该数据集为训练样本训练的模型存储位置）
    if dataset:
        root_data_dir = os.path.abspath(os.path.join(root, "data")) + os.sep
    else:
        root_data_dir = os.path.abspath(os.path.join(root, "data" + os.sep + "{}".format(dataset))) + os.sep

    # 数据目录
    data_dir = os.path.abspath(os.path.join(root_data_dir, "data")) + os.sep

    # 模型目录
    model_dir = os.path.abspath(os.path.join(root_data_dir, "model/" + model_name)) + os.sep

    logger = config_logging(logger=model_name, console_log_level=LogLevel.INFO)
    time_stamp = False

    # 优化器设置
    optimizer = 'adam'
    optimizer_params = {
        'learning_rate': 0.01,
        # 'wd': 0.5,
        # 'momentum': 0.01,
        # 'clip_gradient': 5,
        'epsilon': 0.01,
    }

    # 训练参数设置
    begin_epoch = 0
    end_epoch = 100
    batch_size = 32

    # 运行设备
    ctx = cpu()

    # 是否显示网络图
    view_tag = False

    # 更新保存参数，一般需要保持一致
    save_epoch = 1
    train_select = '^(?!.*embedding)'
    save_select = train_select

    # 用户变量

    def __init__(self, params_yaml=None, **kwargs):
        """
        Configuration File, including categories:

        * directory setting, by default:
            项目根目录($project_name/ --> $root)
            +---数据根目录 ($root/data/ or $root/data/$dataset --> $root_data_dir)
                  +---数据目录 ($root_data_dir/data/ --> $data_dir)
                       +---训练及测试数据 (train & test)
                   +---模型根目录 ($root_data_dir/model/ --- model_root_dir)
                       +--- 模型目录($root_data_dir/model/$model_name --> $model_dir)
                          +--- 模型配置文件(.yaml)
                          +--- 模型日志文件(.log)
                          +--- 模型参数文件(.parmas)
        * optimizer setting
        * training parameters
        * equipment
        * visualization setting
        * parameters saving setting
        * user parameters

        Parameters
        ----------
        params_yaml: str
            The path to configuration file which is in yaml format
        kwargs: dict
            Parameters to be reset.
        """
        params = self.class_var
        if params_yaml:
            params.update(self.load(params_yaml=params_yaml))
        params.update(**kwargs)

        for param, value in params.items():
            setattr(self, "%s" % param, value)

        # rebuild relevant directory or file path according to the kwargs
        if 'root_data_dir' not in kwargs:
            if self.dataset:
                self.root_data_dir = os.path.abspath(os.path.join(self.root, "data")) + os.sep
            else:
                self.root_data_dir = os.path.abspath(
                    os.path.join(self.root, "data" + os.sep + "{}".format(self.dataset))
                ) + os.sep
        if 'data_dir' not in kwargs:
            self.data_dir = os.path.abspath(os.path.join(self.root_data_dir, "data")) + os.sep
        if 'model_dir' not in kwargs:
            if hasattr(self, 'time_stamp') and self.time_stamp and hasattr(self, 'model_dir'):
                time_stamp = "_%s" % datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
                self.model_dir = os.path.abspath(
                    os.path.join(self.root_data_dir, self.model_name)) + time_stamp + os.sep
            else:
                self.model_dir = os.path.abspath(os.path.join(self.root_data_dir, self.model_name)) + os.sep
        self.validation_result_file = os.path.abspath(os.path.join(self.model_dir, "result.json"))

    def items(self):
        return {k: v for k, v in vars(self).items() if k not in {'logger'}}

    @staticmethod
    def load(params_yaml, logger=None):
        f = open(params_yaml)
        params = yaml.load(f)
        if 'ctx' in params:
            params['ctx'] = MXCtx.load(params['ctx'])
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
        """
        获取所有设定的参数

        Returns
        -------
        parameters: dict
            all variables used as parameters
        """
        variables = {k: v for k, v in vars(type(self)).items() if
                     not inspect.isroutine(v) and k not in self.excluded_names()}
        return variables

    @property
    def parsable_var(self):
        """
        获取可以进行命令行设定的参数

        Returns
        -------
        store_vars: dict
            可以进行命令行设定的参数
        """
        store_vars = {k: v for k, v in vars(self).items() if k not in {'logger'}}
        if 'ctx' in store_vars:
            store_vars['ctx'] = MXCtx.dump(store_vars['ctx'])
        return store_vars

    @staticmethod
    def directory_check():
        print("root", Parameters.root)
        print("root_data_dir", Parameters.root_data_dir)
        print("data_dir", Parameters.data_dir)
        print("model_dir", Parameters.model_dir)

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
                'class_var', 'parsable_var', 'directory_check'}


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
    # Advise: firstly checkout whether the directory is correctly (step 1) and
    # then generate the paramters configuation file to check the details (step 2)

    # step 1
    Parameters.directory_check()

    # # step 2
    # default_yaml_file = os.path.join(Parameters.model_dir, "parameters.yaml")
    #
    # # 命令行参数配置
    # parser = ParameterParser()
    # kwargs = parser.parse_args()
    # kwargs = parser.parse(kwargs)
    #
    # data_dir = os.path.join(Parameters.root, "data")
    # parameters = Parameters(
    #     **kwargs
    # )
    # parameters.dump(default_yaml_file, override=True)
    # try:
    #     logger = parameters.logger
    #     parameters.load(default_yaml_file, logger=logger)
    #     parameters.logger.info('format check done')
    # except Exception as e:
    #     print("parameters format error, may contain illegal data type")
    #     raise e
