# coding: utf-8
# create by tongshiwei on 2019-8-30
import torch


def set_device(_net, ctx):
    if ctx == "cpu":
        return _net
    else:
        if torch.cuda.device_count() > 1:
            if ":" in ctx:
                ctx_name, device_ids = ctx.split(":")
                assert ctx_name in ["cuda", "gpu"], "the equipment should be 'cpu', 'cuda' or 'gpu', now is %s" % ctx
                device_ids = device_ids.strip().split(",")
                _net = torch.nn.DataParallel(_net, device_ids)
            elif ctx in ["cuda", "gpu"]:
                return torch.nn.DataParallel(_net)
            else:
                raise TypeError("the equipment should be 'cpu', 'cuda' or 'gpu', now is %s" % ctx)
            return _net

    raise TypeError(
        "ctx should be cpu or gpu, use in format like of 'cpu', 'gpu:', 'cuda', 'gpu: 1, 2' or 'cuda: 1,2,3'"
    )
