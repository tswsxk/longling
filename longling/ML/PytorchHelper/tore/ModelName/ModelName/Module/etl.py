# coding: utf-8
# create by tongshiwei on 2019/4/12

from tqdm import tqdm

__all__ = ["extract", "transform", "etl", "pseudo_data_iter"]


# todo: define extract-transform-load process and implement the pesudo data iterator for testing

def pseudo_data_iter(_cfg):
    def pseudo_data_generation():
        # 在这里定义测试用伪数据流
        import random
        random.seed(10)

        raw_data = [
            [random.random() for _ in range(5)]
            for _ in range(1000)
        ]

        return raw_data

    return transform(pseudo_data_generation(), _cfg)


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

    raise NotImplementedError


def etl(*args, params):
    raw_data = extract(*args)
    transformed_data = transform(raw_data, params)
    load(transformed_data, params)
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
