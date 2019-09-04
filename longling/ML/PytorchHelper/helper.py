# coding: utf-8
# create by tongshiwei on 2019-8-30


def pick(tensor, index, axis=-1):
    return torch.gather(tensor, axis, index.unsqueeze(axis)).squeeze(axis)
