# coding: utf-8
# create by tongshiwei on 2020-11-16
import os
from longling import ConfigurationParser

__all__ = ["Configuration", "ConfigurationParser", "directory_check"]

import datetime

from longling import path_append
from longling.lib import parser
from longling.lib.parser import var2exp
from longling.lib.utilog import config_logging, LogLevel
from ..const import CFG_JSON, RESULT_JSON


class Configuration(parser.Configuration):
    # 目录配置
    model_name = "MLModel"

    root = "./"
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

        Notes
        -----
        Developers modifying this code should simultaneously modify the relevant codes in glue and tore

        Examples
        --------
        >>> cfg = Configuration(epoch=5)
        >>> cfg["epoch"]
        5
        >>> cfg   # doctest: +ELLIPSIS
        logger: <Logger MLModel (INFO)>
        model_name: MLModel
        root: ./
        ...
        """
        super(Configuration, self).__init__()

        params = self.class_var
        if params_path:
            params.update(self.load_cfg(cfg_path=params_path, **(params_kwargs if params_kwargs else {})))
        params.update(**kwargs)

        self.validation_result_file = None
        self.cfg_path = None

        self._update(**params)

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
            kwargs["model_dir"] = path_append("$root_model_dir", "$workspace")

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


def directory_check(class_obj):
    print("root", os.path.abspath(class_obj.root))


if __name__ == '__main__':
    # Advise: firstly checkout whether the directory is correctly (step 1) and
    # then generate the paramters configuation file
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
