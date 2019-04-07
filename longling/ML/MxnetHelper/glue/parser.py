# coding:utf-8
# created by tongshiwei on 2018/8/8
from mxnet import Context
from mxnet import cpu, gpu

from longling.lib.candylib import as_list


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
