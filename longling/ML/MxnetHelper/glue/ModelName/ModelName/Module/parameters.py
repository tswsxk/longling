# coding: utf-8
# Copyright @tongshiwei

from __future__ import absolute_import
from __future__ import print_function

import datetime
import pathlib

from mxnet import cpu

import longling.ML.MxnetHelper.glue.parser as parser
from longling.ML.MxnetHelper.glue.parser import path_append, var2exp
from longling.ML.MxnetHelper.toolkit.optimizer_cfg import get_optimizer_cfg
from longling.lib.utilog import config_logging, LogLevel


class Parameters(parser.Parameters):
    # 目录配置
    model_name = str(pathlib.Path(__file__).parents[1].name)

    root = pathlib.Path(__file__).parents[2]
    dataset = ""
    time_stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    workspace = ""

    root_data_dir = path_append(root / "data", dataset)
    data_dir = path_append(
        root_data_dir, "data", to_str=True
    )
    model_dir = path_append(
        root_data_dir / "model" / model_name, workspace, to_str=True
    )

    root = str(root)
    root_data_dir = str(root_data_dir)

    # 优化器设置
    optimizer, optimizer_params = get_optimizer_cfg(
        name="base",
        lr_step=1000,
        lr_total_step=10000,
    )

    # 训练参数设置
    begin_epoch = 0
    end_epoch = 100
    batch_size = 32
    save_epoch = 1
    # 更新保存参数，一般需要保持一致
    train_select = '^(?!.*embedding)'
    save_select = train_select

    # 运行设备
    ctx = cpu()

    # 用户变量

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
        super(Parameters, self).__init__(
            logger=config_logging(
                logger=self.model_name,
                console_log_level=LogLevel.INFO
            )
        )

        params = self.class_var
        if params_json:
            params.update(self.load(params_json=params_json))
        params.update(**kwargs)

        for param, value in params.items():
            setattr(self, "%s" % param, value)

        # rebuild relevant directory or file path according to the kwargs
        _dirs = ["root_data_dir", "data_dir", "model_dir"]
        for _dir in _dirs:
            setattr(self, _dir, eval(var2exp(
                kwargs.get(_dir, getattr(self, _dir))
            )))

        self.validation_result_file = path_append(
            self.model_dir, "result.json", to_str=True
        )


def directory_check(obj):
    print("data_dir", obj.data_dir)
    print("model_dir", obj.model_dir)


if __name__ == '__main__':
    # Advise: firstly checkout whether the directory is correctly (step 1) and
    # then generate the paramters configuation file
    # to check the details (step 2)

    # step 1
    directory_check(Parameters)

    # # step 2
    default_parameters_file = path_append(
        Parameters.model_dir, "parameters.json", to_str=True
    )

    # 命令行参数配置
    parser = parser.ParameterParser(Parameters)
    kwargs = parser.parse_args()
    kwargs = parser.parse(kwargs)

    parameters = Parameters(
        **kwargs
    )
    parameters.dump(default_parameters_file, override=True)
    try:
        logger = parameters.logger
        parameters.load(default_parameters_file)
        parameters.logger = logger
        parameters.logger.info('format check done')
    except Exception as e:
        print("parameters format error, may contain illegal data type")
        raise e
