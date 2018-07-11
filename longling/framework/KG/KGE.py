# coding:utf-8
# created by tongshiwei on 2018/7/9

"""
测试文件
"""
from longling.framework.KG.dataset.construction import sro_jsonxz, pair_jsonxz, full_jsonxz

from multiprocessing import Pool


def FB15():
    root = "../../data/KG/FB15/"
    train_file = root + "freebase_mtr100_mte100-train.txt"
    test_file = root + "freebase_mtr100_mte100-test.txt"
    valid_file = root + "freebase_mtr100_mte100-valid.txt"
    pool = Pool()
    pool.apply_async(pair_jsonxz, args=(train_file, root + "train.jsonxz",))
    pool.apply_async(full_jsonxz, args=(test_file, root + "test.jsonxz",))
    pool.apply_async(sro_jsonxz, args=(valid_file, root + "valid.jsonxz",))
    pool.close()
    pool.join()
    # full_jsonxz(test_file, root + "test.jsonxz")


if __name__ == '__main__':
    FB15()

