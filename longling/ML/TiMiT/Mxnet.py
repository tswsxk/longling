# coding: utf-8
# create by tongshiwei on 2020-11-12

import functools

from longling.ML.MxnetHelper.glue import MetaModule, MetaModel, fit_f as meta_fit_f

__all__ = ["get_mx_module", "get_mx_model", "to_fit_f"]


def to_fit_f(fit_step_func):
    @functools.wraps(meta_fit_f)
    def fit_f(*args, **kwargs):
        return meta_fit_f(*args, fit_step_func=fit_step_func, **kwargs)

    return fit_f


def get_mx_module(
        module_cls: type(MetaModule) = MetaModule,
        net_gen_func=None,
        fit_func=None,
        net_init_func=None,
        eval_func=None
) -> type(MetaModule):
    """

    Parameters
    ----------
    module_cls
    net_gen_func
    fit_func
    net_init_func
    eval_func

    Returns
    -------

    """
    net_gen_func = module_cls.sym_gen if net_gen_func is None else net_gen_func
    fit_func = module_cls.fit_f if fit_func is None else fit_func
    net_init_func = module_cls.net_initialize if net_init_func is None else net_init_func
    eval_func = module_cls.eval if eval_func is None else eval_func

    class MetaModule(module_cls):
        @functools.wraps(net_gen_func)
        def sym_gen(self, *args, **kwargs):
            return net_gen_func(*args, **kwargs)

        @functools.wraps(fit_func)
        def fit_f(self, *args, **kwargs):
            return fit_func(*args, **kwargs)

        @functools.wraps(net_init_func)
        def net_initialize(self, *args, **kwargs):
            return net_init_func(*args, **kwargs)

        @staticmethod
        @functools.wraps(eval_func)
        def eval(*args, **kwargs):
            return eval_func(*args, **kwargs)

    return MetaModule


def get_mx_model(
        model_cls: type(MetaModel) = MetaModel,
        module_cls=None,
        configuration_cls=None,
        configuration_parser_cls=None,
) -> type(MetaModel):
    module_cls = MetaModel.get_module_cls() if module_cls is not None else model_cls
    configuration_cls = MetaModel.get_configuration_cls() if configuration_cls is not None else configuration_cls
    configuration_parser_cls = MetaModel.get_configuration_parser_cls(
    ) if configuration_parser_cls is not None else configuration_parser_cls

    class Model(model_cls):
        @staticmethod
        def get_module_cls():
            return module_cls

        @staticmethod
        def get_configuration_cls():
            return configuration_cls

        @staticmethod
        def get_configuration_parser_cls():
            return configuration_parser_cls

    return Model
