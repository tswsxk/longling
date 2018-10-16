# coding: utf-8
# Copyright @tongshiwei

try:
    from .GluonModule import Parameters, GluonModule
except (SystemError, ModuleNotFoundError):
    from GluonModule import Parameters, GluonModule


def load_net(load_epoch=Parameters.end_epoch, params=Parameters()):
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
    def __init__(self, end_epoch=Parameters.end_epoch, params=Parameters()):
        self.net = load_net(end_epoch, params)

    def __call__(self, *args, **kwargs):
        pass
