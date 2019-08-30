# coding: utf-8
# create by tongshiwei on 2019/4/12

import mxnet as mx
from mxnet import gluon
from tqdm import tqdm

__all__ = ["extract", "transform", "etl"]


def extract(data_src):
    raise NotImplementedError


def transform(raw_data, params):
    # 定义数据转换接口
    # raw_data --> batch_data

    batch_size = params.batch_size

    transformed_data = raw_data

    return transformed_data


def load(transformed_data, params):
    batch_size = params.batch_size

    return gluon.data.DataLoader(
        gluon.data.ArrayDataset(
            mx.nd.array(transformed_data, dtype="float32")
        ),
        batch_size
    )


def etl(*args, params):
    raw_data = extract(*args)
    transform(raw_data, params)
    raise NotImplementedError


if __name__ == '__main__':
    from longling.lib.structure import AttrDict
    import os

    filename = "../../data/data.json"
    print(os.path.abspath(filename))

    for data in tqdm(extract(filename)):
        pass

    parameters = AttrDict({"batch_size": 128})
    for data in tqdm(etl(filename, params=parameters)):
        pass
