# coding: utf-8
# Copyright @tongshiwei

from __future__ import absolute_import
from __future__ import print_function

import datetime
import pathlib

from mxnet import cpu

import longling.ML.MxnetHelper.glue.parser as parser
from longling.ML.MxnetHelper.glue.parser import path_append, var2exp
from longling.ML.MxnetHelper.toolkit.optimizer_cfg import get_optimizer_cfg, \
    get_update_steps
from longling.ML.MxnetHelper.toolkit.select_exp import all_params as _select
from longling.lib.utilog import config_logging, LogLevel


class Configuration(parser.Configuration):
    # 目录配置
    model_name = str(pathlib.Path(__file__).parents[1].name)

    root = pathlib.Path(__file__).parents[2]
    dataset = ""
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    workspace = ""

    root_data_dir = path_append(root / "data", dataset)
    data_dir = path_append(
        root_data_dir, "data", to_str=True
    )
    root_model_dir = root_data_dir / "model" / model_name
    model_dir = path_append(
        root_model_dir, workspace, to_str=True
    )
    cfg_path = path_append(
        model_dir, "configuration.json", to_str=True
    )

    root = str(root)
    root_data_dir = str(root_data_dir)
    root_model_dir = str(root_model_dir)

    # 优化器设置
    optimizer, optimizer_params = get_optimizer_cfg(name="base")
    lr_params = {
        "step": 100,
        "max_update_steps": get_update_steps(
            update_epoch=10,
            batches_per_epoch=1000,
        ),
    }

    # 训练参数设置
    begin_epoch = 0
    end_epoch = 100
    batch_size = 32
    save_epoch = 1
    # 更新保存参数，一般需要保持一致
    train_select = _select
    save_select = train_select

    # 运行设备
    ctx = cpu()

    # 用户变量
    # 超参数
    hyper_params = {}

    def __init__(self, params_json=None, **kwargs):
        """
        Configuration File, including categories:

        * directory setting, by default:
        * optimizer setting
        * training parameters
        * equipment
        * visualization setting
        * parameters saving setting
        * user parameters

        Parameters
        ----------
        params_json: str
            The path to configuration file which is in json format
        kwargs: dict
            Parameters to be reset.
        """
        super(Configuration, self).__init__(
            logger=config_logging(
                logger=self.model_name,
                console_log_level=LogLevel.INFO
            )
        )

        params = self.class_var
        if params_json:
            params.update(self.load(cfg_path=params_json))
        params.update(**kwargs)

        for param, value in params.items():
            setattr(self, "%s" % param, value)

        # set workspace
        if kwargs.get("workspace"):
            kwargs["model_dir"] = path_append(
                "$root_model_dir", "$workspace", to_str=True
            )

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

        self.validation_result_file = path_append(
            self.model_dir, "result.json", to_str=True
        )
        self.cfg_path = path_append(
            self.model_dir, "configuration.json", to_str=True
        )

    def dump(self, cfg_path=None, override=False):
        cfg_path = self.cfg_path if cfg_path is None else cfg_path
        super(Configuration, self).dump(cfg_path, override)

    @staticmethod
    def load(cfg_path=None):
        cfg_path = Configuration.cfg_path if cfg_path is None else cfg_path
        return parser.Configuration.load(cfg_path)


class ConfigurationParser(parser.ConfigurationParser):
    pass


def directory_check(class_obj):
    print("data_dir", class_obj.data_dir)
    print("model_dir", class_obj.model_dir)


if __name__ == '__main__':
    # Advise: firstly checkout whether the directory is correctly (step 1) and
    # then generate the paramters configuation file
    # to check the details (step 2)

    # step 1
    directory_check(Configuration)

    # 命令行参数配置
    kwargs = ConfigurationParser.get_cli_cfg(Configuration)

    cfg = Configuration(
        **kwargs
    )
    print(kwargs)
    print(cfg)

    # # step 2
    cfg.dump(override=True)
    try:
        logger = cfg.logger
        cfg.load()
        cfg.logger = logger
        cfg.logger.info('format check done')
    except Exception as e:
        print("parameters format error, may contain illegal data type")
        raise e
