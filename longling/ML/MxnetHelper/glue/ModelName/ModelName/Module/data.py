# coding: utf-8
# create by tongshiwei on 2019/4/12

from mxnet import gluon
from tqdm import tqdm

__all__ = ["load_data", "transform", "get_data_iter"]


def load_data(data_src):
    raise NotImplementedError


def transform(raw_data, params):
    # 定义数据转换接口
    # raw_data --> batch_data

    batch_size = params.batch_size

    return gluon.data.DataLoader(gluon.data.ArrayDataset(raw_data), batch_size)


def get_data_iter(*args, **kwargs):
    raw_data = load_data(*args)
    transform(raw_data, **kwargs)
    raise NotImplementedError


if __name__ == '__main__':
    for data in tqdm(get_data_iter()):
        pass
