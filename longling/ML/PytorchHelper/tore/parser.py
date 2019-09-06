# coding: utf-8
# create by tongshiwei on 2019-9-6

from longling.lib.parser import get_parsable_var, load_configuration_json, \
    var2exp, path_append, Configuration as Params, ConfigurationParser

__all__ = [
    "Configuration",
    "var2exp", "eval_var",
    "path_append", "ConfigurationParser",
]


def eval_var(variable):
    return eval(variable)


class Configuration(Params):
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
            parse_exclude={'logger'},
        )

    @staticmethod
    def load_cfg(params_json, **kwargs):
        with open(params_json) as f:
            params = load_configuration_json(
                f,
            )
        params.update(kwargs)
        return params

    @staticmethod
    def load(cfg_path, **kwargs):
        Configuration(Configuration.load_cfg(cfg_path, **kwargs))
