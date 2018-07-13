# coding:utf-8
# created by tongshiwei on 2018/7/9

"""
测试文件
"""
from longling.framework.KG.io_lib import ERMapper
from longling.framework.KG.dataset.construction import sro_jsonxz, pair_jsonxz, full_jsonxz

from multiprocessing import Pool


def FB15():
    root = "../../data/KG/FB15/"
    raw_train_file = root + "freebase_mtr100_mte100-train.txt"
    raw_test_file = root + "freebase_mtr100_mte100-test.txt"
    raw_valid_file = root + "freebase_mtr100_mte100-valid.txt"

    entities_index = root + "freebase_mtr100_mte100-entities.txt"
    relations_index = root + "freebase_mtr100_mte100-relations.txt"

    train_file = root + "freebase_mtr100_mte100-train"
    test_file = root + "freebase_mtr100_mte100-test"
    valid_file = root + "freebase_mtr100_mte100-valid"

    mapper = ERMapper(base_filename=raw_train_file)
    mapper.save(entities_index, relations_index)

    pool = Pool()
    pool.apply_async(mapper.transform, args=(raw_train_file, train_file))
    pool.apply_async(mapper.transform, args=(raw_test_file, test_file))
    pool.apply_async(mapper.transform, args=(raw_valid_file, valid_file))

    pool.close()
    pool.join()

    pool = Pool()
    pool.apply_async(pair_jsonxz, args=(train_file, root + "train.jsonxz",))
    # pool.apply_async(full_jsonxz, args=(test_file, root + "test.jsonxz", None, 100))
    pool.apply_async(sro_jsonxz, args=(valid_file, root + "valid.jsonxz",))
    # pool.apply_async(full_jsonxz,
    #                  args=(test_file, root + "big_test.jsonxz", [train_file, test_file, valid_file], 14951, ))
    # pool.apply_async(full_jsonxz, args=(test_file, root + "full_test.jsonxz", [train_file, test_file, valid_file]))
    pool.close()
    pool.join()

    full_jsonxz(test_file, root + "test.jsonxz", negtive_ratio=15)
    full_jsonxz(test_file, root + "big_test.jsonxz", [train_file, test_file, valid_file], 5000,)
    full_jsonxz(test_file, root + "full_test.jsonxz", [train_file, test_file, valid_file])


if __name__ == '__main__':
    FB15()

