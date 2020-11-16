# coding: utf-8
# Copyright @tongshiwei
from __future__ import absolute_import
from __future__ import print_function

__all__ = ["Configuration", "ConfigurationParser"]

import datetime
import pathlib

from mxnet import cpu

from longling import path_append
from longling.ML.MxnetHelper.glue import parser
from longling.ML.MxnetHelper.glue.parser import eval_var
from longling.ML.MxnetHelper.toolkit import get_optimizer_cfg
from longling.ML.MxnetHelper.toolkit.select_exp import all_params as _select
from longling.lib.parser import var2exp
from longling.lib.utilog import config_logging, LogLevel


class Configuration(parser.Configuration):
    # 目录配置
    model_name = str(pathlib.Path(__file__).parents[1].name)

    root = pathlib.Path(__file__).parents[2]
    dataset = ""
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    workspace = ""

    root_data_dir = path_append("$root", "data", "$dataset") if dataset else path_append("$root", "data")
    data_dir = path_append("$root_data_dir", "data")
    root_model_dir = path_append("$root_data_dir", "model", "$model_name")
    model_dir = path_append("$root_model_dir", "$workspace") if workspace else root_model_dir
    cfg_path = path_append("$model_dir", "configuration.json")

    root = str(root)
    root_data_dir = str(root_data_dir)
    root_model_dir = str(root_model_dir)

    # 训练参数设置
    begin_epoch = 0
    end_epoch = 100
    batch_size = 32
    save_epoch = 1

    # 优化器设置
    optimizer, optimizer_params = get_optimizer_cfg(name="base")
    # lr_params 中包含 update_params 时，学习率衰减方式运行时确定
    lr_params = {"update_params": {}}

    # 更新保存参数，一般需要保持一致
    train_select = _select
    save_select = train_select

    # 运行设备
    ctx = cpu()

    # 工具包参数
    toolbox_params = {}

    # 用户变量
    # 网络超参数
    hyper_params = {}
    # 网络初始化参数
    init_params = {}
    # 损失函数超参数
    loss_params = {}

    # 说明
    caption = ""

    def __init__(self, params_path=None, **kwargs):
        """
        Configuration File, including categories:

        * directory setting
        * optimizer setting
        * training parameters
        * equipment
        * parameters saving setting
        * user parameters

        Parameters
        ----------
        params_path: str
            The path to configuration file which is in json format
        kwargs:
            Parameters to be reset.
        """
        super(Configuration, self).__init__(
            logger=config_logging(
                logger=self.model_name,
                console_log_level=LogLevel.INFO
            )
        )

        params = self.class_var
        if params_path:
            params.update(self.load_cfg(cfg_path=params_path))
        params.update(**kwargs)

        for key in params:
            if key.endswith("_params") and key + "_update" in params:
                params[key].update(params[key + "_update"])

        # path_override_check
        path_check_list = ["dataset", "root_data_dir", "workspace", "root_model_dir", "model_dir"]
        _overridden = {}
        for path_check in path_check_list:
            if kwargs.get(path_check) is None or kwargs[path_check] == getattr(self, "%s" % path_check):
                _overridden[path_check] = False
            else:
                _overridden[path_check] = True

        for param, value in params.items():
            setattr(self, "%s" % param, value)

        def is_overridden(varname):
            return _overridden["%s" % varname]

        # set dataset
        if is_overridden("dataset") and not is_overridden("root_data_dir"):
            kwargs["root_data_dir"] = path_append("$root", "data", "$dataset")
        # set workspace
        if (is_overridden("workspace") or is_overridden("root_model_dir")) and not is_overridden("model_dir"):
            kwargs["model_dir"] = path_append("$root_model_dir", "workspace")

        # rebuild relevant directory or file path according to the kwargs
        _dirs = [
            "workspace", "root_data_dir", "data_dir", "root_model_dir",
            "model_dir"
        ]
        for _dir in _dirs:
            exp = var2exp(
                kwargs.get(_dir, getattr(self, _dir)),
                env_wrap=lambda x: "self.%s" % x
            )
            setattr(self, _dir, eval(exp))

        _vars = [
            "ctx"
        ]
        for _var in _vars:
            if _var in kwargs:
                try:
                    setattr(self, _var, eval_var(kwargs[_var]))
                except TypeError:
                    pass

        self.validation_result_file = path_append(
            self.model_dir, "result.json", to_str=True
        )
        self.cfg_path = path_append(
            self.model_dir, "configuration.json", to_str=True
        )

    def dump(self, cfg_path=None, override=True, file_format=None):
        cfg_path = self.cfg_path if cfg_path is None else cfg_path
        super(Configuration, self).dump(cfg_path, override, file_format=file_format)

    def var2val(self, var):
        return eval(var2exp(
            var,
            env_wrap=lambda x: "self.%s" % x
        ))


class ConfigurationParser(parser.ConfigurationParser):
    pass


def directory_check(class_obj):
    import os
    print("root", os.path.abspath(class_obj.root))


if __name__ == '__main__':
    # Advise: firstly checkout whether the directory is correctly (step 1) and
    # then generate the parameters configuration file
    # to check the details (step 2)
    stage = 1
    # step 1
    directory_check(Configuration)

    if stage > 1:
        # 命令行参数配置
        _kwargs = ConfigurationParser.get_cli_cfg(Configuration)

        cfg = Configuration(
            **_kwargs
        )
        print(_kwargs)
        print(cfg)

        if stage > 2:
            # # step 2
            cfg.dump(override=True)
            try:
                logger = cfg.logger
                cfg.load_cfg(cfg.cfg_path)
                cfg.logger = logger
                cfg.logger.info('format check done')
            except Exception as e:
                print("parameters format error, may contain illegal data type")
                raise e
