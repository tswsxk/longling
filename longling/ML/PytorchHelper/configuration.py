# coding: utf-8
# 2021/3/13 @ tongshiwei

__all__ = ["Configuration", "ConfigurationParser"]

from longling import path_append
from longling.ML.PytorchHelper.tore import parser
from longling.ML.PytorchHelper.toolkit import get_optimizer_cfg
from longling.ML.DL import ALL_PARAMS as SELECT
from longling.lib.parser import var2exp
from longling.lib.utilog import config_logging
from longling.ML.const import RESULT_JSON, CFG_JSON


class Configuration(parser.Configuration):
    model_name = "model"
    model_dir = model_name

    # 训练参数设置
    begin_epoch = 0
    end_epoch = 100
    batch_size = 32
    save_epoch = 1

    # 优化器设置
    optimizer, optimizer_params = get_optimizer_cfg(name="base")
    # lr_params 中包含 update_params 时，学习率衰减方式运行时确定
    # lr_params = {"update_params": {}}
    lr_params = {}

    # 更新保存参数，一般需要保持一致
    train_select = SELECT
    save_select = train_select

    # 运行设备
    ctx = "cpu"
    train_ctx = None
    eval_ctx = None

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

    def __init__(self, params_path=None, params_kwargs=None, **kwargs):
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

        Examples
        -------
        >>> cfg = Configuration(model_name="longling", model_dir="./")
        >>> cfg  # doctest: +ELLIPSIS
        logger: <Logger longling (INFO)>
        model_name: longling
        model_dir: ./
        ...
        >>> cfg.var2val("$model_dir")
        './'
        """
        super(Configuration, self).__init__()
        params = self.class_var
        if params_path:
            params.update(self.load_cfg(cfg_path=params_path, **(params_kwargs if params_kwargs else {})))
        params.update(**kwargs)

        self.validation_result_file = None
        self.cfg_path = None

        self.update(**params)

    def _update(self, **kwargs):
        params = kwargs
        params["logger"] = params.pop(
            "logger",
            config_logging(
                logger=params.get("model_name", self.model_name),
                console_log_level="info"
            )
        )

        for key in params:
            if key.endswith("_params") and key + "_update" in params:
                params[key].update(params[key + "_update"])

        self.deep_update(**params)

        self.validation_result_file = path_append(
            self.model_dir, RESULT_JSON, to_str=True
        )
        self.cfg_path = path_append(
            self.model_dir, CFG_JSON, to_str=True
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
