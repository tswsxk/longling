# coding:utf-8
# created by tongshiwei on 2018/8/8
from mxnet import Context, cpu, gpu

from longling.lib.candylib import as_list
from longling.lib.parser import get_parsable_var, var2exp, path_append, Configuration as Params, ConfigurationParser

__all__ = [
    "MXCtx", "Configuration",
    "var2exp", "eval_var",
    "path_append", "ConfigurationParser",
    "cpu", "gpu"
]


def eval_var(variable):
    return eval(variable)


class MXCtx(object):
    @staticmethod
    def load(data):
        ctx_vars = []
        for device_type, device_ids in data.items():
            if isinstance(device_ids, int):
                device_ids = as_list(device_ids)
            elif isinstance(device_ids, str):
                device_ids = map(int, device_ids.split(','))
            for device_id in device_ids:
                ctx_vars.append(eval(device_type)(device_id))
        return ctx_vars

    @staticmethod
    def dump(data):
        ctx_vars = {}
        for ctx in as_list(data):
            assert isinstance(ctx, Context)
            if ctx.device_type not in ctx_vars:
                ctx_vars[ctx.device_type] = []
            ctx_vars[ctx.device_type].append(ctx.device_id)
        for device_type, device_ids in ctx_vars.items():
            if len(device_ids) > 1:
                ctx_vars[device_type] = ",".join(list(map(str, device_ids)))
            else:
                ctx_vars[device_type] = device_ids[0]
        return ctx_vars


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
            parse_exclude=None,
            dump_parse_functions={'ctx': MXCtx.dump}
        )

    @classmethod
    def load_parse_function(cls):
        return {"ctx": MXCtx.load}


if __name__ == '__main__':
    print(eval_var(
        "[gpu(0), gpu(1), gpu(2)]"
    ))
