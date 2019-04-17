# coding:utf-8
# created by tongshiwei on 2018/7/9

"""
测试文件
"""
import json
from multiprocessing import Pool

from dev.framework.KG.base import logger
from dev.framework.KG.io_lib import ERMapper, UniteMapper, SROMapper


def FB15(map_type='er', test_num=100, big_test_num=5000, full_tag=True, verify_tag=True, link_class=True,
         link_prediction=True, ):
    # full test 需要至少35G空间
    root = "../../data/KG/FB15/"
    raw_train_file = root + "freebase_mtr100_mte100-train.txt"
    raw_test_file = root + "freebase_mtr100_mte100-test.txt"
    raw_valid_file = root + "freebase_mtr100_mte100-valid.txt"

    assert map_type in ('er', 'u', 'sro')
    dataset_dir = root + map_type + '/'

    # build map dict
    if map_type == 'er':
        entities_map_filename = dataset_dir + "freebase_mtr100_mte100-entities.txt"
        relations_map_filename = dataset_dir + "freebase_mtr100_mte100-relations.txt"
        mapper = ERMapper(base_filename=raw_train_file)
        mapper.save(entities_map_filename, relations_map_filename)
    elif map_type == 'u':
        unite_map_filename = dataset_dir + "freebase_mtr100_mte100-unite.txt"
        mapper = UniteMapper(base_filename=raw_train_file)
        mapper.save(unite_map_filename)
    else:
        subjects_map_filename = dataset_dir + "freebase_mtr100_mte100-subjects.txt"
        relations_map_filename = dataset_dir + "freebase_mtr100_mte100-relations.txt"
        objects_map_filename = dataset_dir + "freebase_mtr100_mte100-objects.txt"
        mapper = SROMapper(base_filename=raw_train_file)
        mapper.save(subjects_map_filename, relations_map_filename, objects_map_filename)

    train_file = dataset_dir + "train"
    test_file = dataset_dir + "freebase_mtr100_mte100-test"
    valid_file = dataset_dir + "freebase_mtr100_mte100-valid"

    # data transform, from string to index
    pool = Pool()
    pool.apply_async(mapper.transform, args=(raw_train_file, train_file))
    pool.apply_async(mapper.transform, args=(raw_test_file, test_file))
    pool.apply_async(mapper.transform, args=(raw_valid_file, valid_file))
    pool.close()
    pool.join()

    if verify_tag:
        pool = Pool()
        pool.apply_async(verify_data, args=(train_file,))
        pool.apply_async(verify_data, args=(test_file,))
        pool.apply_async(verify_data, args=(valid_file,))
        pool.close()
        pool.join()

    # relation classification
    if link_class:
        one2one_filename, one2many_filename, many2one_filename, many2many_filename = \
            relation_classification(test_file, test_file, sources=[train_file, test_file, valid_file])
        if verify_tag:
            pool = Pool()
            pool.apply_async(verify_data, args=(one2one_filename,))
            pool.apply_async(verify_data, args=(one2many_filename,))
            pool.apply_async(verify_data, args=(many2one_filename,))
            pool.apply_async(verify_data, args=(many2many_filename,))
            pool.close()
            pool.join()

    # relation prediction
    if link_prediction:
        n_rel_prefix = dataset_dir + "rp_n_rel"
        n_rest_prefix = dataset_dir + "rp_n_rest"
        n_rel_train, n_rel_test, n_rest_train, n_rest_test = relation_prediction([train_file, test_file, valid_file],
                                                                                 n_rel_prefix,
                                                                                 n_rest_prefix)
        if verify_tag:
            pool = Pool()
            pool.apply_async(verify_data, args=(n_rel_train,))
            pool.apply_async(verify_data, args=(n_rel_test,))
            pool.apply_async(verify_data, args=(n_rest_train,))
            pool.apply_async(verify_data, args=(n_rest_test,))
            pool.close()
            pool.join()
    # generate test file
    if test_num:
        filename = dataset_dir + "test.jsonxz"
        full_jsonxz(test_file, filename, negative_ratio=test_num)
        if verify_tag:
            verify_data(filename, 'jsonxz')
    if big_test_num:
        filename = dataset_dir + "big_test.jsonxz"
        full_jsonxz(test_file, filename, [train_file, test_file, valid_file], big_test_num)
        if verify_tag:
            verify_data(filename, 'jsonxz')
    if full_tag:
        filename = dataset_dir + "full_test.jsonxz"
        full_jsonxz(test_file, filename, [train_file, test_file, valid_file])
        if verify_tag:
            verify_data(filename, 'jsonxz')


def build_dataset(dataset_name, map_type, **kwargs):
    eval(dataset_name)(map_type, **kwargs)


def verify_data(filename, file_type=''):
    from tqdm import tqdm
    line_checker = json.loads if 'json' in file_type else lambda x: x
    with open(filename) as f:
        for i, line in tqdm(enumerate(f), "examine %s" % filename):
            try:
                line_checker(line)
            except Exception as e:
                logger.error("%s\n error happen in %s line\n %s\n" % (e, i + 1, line))
    logger.info("%s verification completed, pass")


if __name__ == '__main__':
    FB15()