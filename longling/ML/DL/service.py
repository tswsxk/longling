# coding: utf-8
# 2020/1/7 @ tongshiwei


class ServiceModule(object):  # pragma: no cover
    def __init__(self, cfg=None, **kwargs):
        self.cfg = self.config(cfg, **kwargs)
        self.mod = self.get_module(self.cfg)

        self.loss_function = None

    @staticmethod
    def get_configuration_cls():
        raise NotImplementedError

    @staticmethod
    def get_module_cls():
        raise NotImplementedError

    @classmethod
    def config(cls, cfg=None, **kwargs):
        """
        配置初始化

        Parameters
        ----------
        cfg: Configuration, str or None
            默认为 None，不为None时，将使用cfg指定的参数配置
            或路径指定的参数配置作为模型参数
        kwargs
            参数配置可选参数
        """
        configuration_cls = cls.get_configuration_cls()

        cfg = configuration_cls(
            **kwargs
        ) if cfg is None else cfg
        if not isinstance(cfg, configuration_cls):
            cfg = configuration_cls.load_cfg(cfg, **kwargs)
        return cfg

    @classmethod
    def get_module(cls, cfg):
        """
        根据配置，生成模型模块

        Parameters
        ----------
        cfg: Configuration
            模型配置参数
        Returns
        -------
        mod: Module
            模型模块
        """
        module_cls = cls.get_module_cls()

        mod = module_cls(cfg)
        mod.logger.info(str(mod))
        filename = mod.cfg.cfg_path
        mod.logger.info("parameters saved to %s" % filename)
        return mod


class CliServiceModule(ServiceModule):  # pragma: no cover
    @staticmethod
    def get_configuration_cls():
        raise NotImplementedError

    @staticmethod
    def get_module_cls():
        raise NotImplementedError

    @staticmethod
    def get_configuration_parser_cls():
        raise NotImplementedError

    @classmethod
    def cli_commands(cls):
        raise NotImplementedError

    @classmethod
    def get_parser(cls):
        configuration_parser_cls = cls.get_configuration_parser_cls()
        configuration_cls = cls.get_configuration_cls()

        cfg_parser = configuration_parser_cls(
            configuration_cls,
            commands=cls.cli_commands()
        )
        return cfg_parser

    @classmethod
    def run(cls, parse_args=None):
        cfg_parser = cls.get_parser()
        cfg_kwargs = cfg_parser(parse_args)

        if "subcommand" not in cfg_kwargs:
            cfg_parser.print_help()
            return
        subcommand = cfg_kwargs["subcommand"]
        del cfg_kwargs["subcommand"]

        eval("cls.%s" % subcommand)(**cfg_kwargs)


def service_wrapper(
        meta_model_cls: type(CliServiceModule),
        configuration_cls=None,
        configuration_parser_cls=None,
        module_cls=None):  # pragma: no cover
    class MetaModel(meta_model_cls):
        @staticmethod
        def get_configuration_cls():
            if configuration_cls is not None:
                return configuration_cls
            else:
                return meta_model_cls.get_configuration_cls()

        @staticmethod
        def get_module_cls():
            if module_cls is not None:
                return module_cls
            else:
                return meta_model_cls.get_module_cls()

        @staticmethod
        def get_configuration_parser_cls():
            if configuration_parser_cls is not None:
                return configuration_parser_cls
            else:
                return meta_model_cls.get_configuration_parser_cls()

    return MetaModel
