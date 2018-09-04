# coding: utf-8
# Copyright @tongshiwei

from .GluonModule import Parameters, GluonModule


def load_net(load_epoch=Parameters.end_epoch, **kwargs):
    params = Parameters(
        **kwargs
    )

    mod = GluonModule(params)

    net = mod.sym_gen()
    net = mod.load(net, load_epoch, mod.params.ctx)
    return net


# todo 重命名eval_module_name函数到需要的模块名
def eval_module_name(load_epoch):
    net = load_net()
    pass


# todo 重命名use_module_name函数到需要的模块名
class module_name(object):
    def __init__(self, load_epoch):
        self.net = load_net()

    def __call__(self, *args, **kwargs):
        pass
