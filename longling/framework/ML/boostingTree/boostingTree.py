# coding: utf-8
# create by tongshiwei on 2017/12/5
from __future__ import unicode_literals
from __future__ import print_function

from abc import ABCMeta, abstractmethod

import numpy as np
import codecs
from catboost import Pool, CatBoostClassifier


class boostingTree(object):
    __metaclass__ = ABCMeta

    def __init__(self, collaborate_feature_num=30, **kwargs):
        self.base_model = self.get_base_model()
        self.feature_models = []
        self.model = None
        self.kwargs = kwargs
        self.collaborate_feature_num = collaborate_feature_num
        self.kwargs['n_estimators'] = self.collaborate_feature_num

    @abstractmethod
    def get_base_model(self):
        pass

    def partial_fit(self, X, y):
        if self.model is not None:
            self.feature_models.append(self.model)
        X = self._apply(X)
        self.model = self._fit(X, y)

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        model = self.base_model(**self.kwargs)
        model.fit(X, y)
        self.model = model

    def predict(self, X):
        return self.model.predict(self._apply(X))

    def apply(self, X, only_leaves=True):
        new_f = self.model.apply(self._apply(X))
        if only_leaves:
            return new_f
        return np.hstack((X, np.reshape(new_f, (new_f.shape[0], new_f.shape[1]))))

    def _apply(self, X):
        for model in self.feature_models:
            new_f = model.apply(X)
            X = np.hstack((X, np.reshape(new_f, (new_f.shape[0], new_f.shape[1]))))
        return X

    def _fit(self, X, y):
        model = self.base_model(**self.kwargs)
        model.fit(X, y)
        return model


class Xgboost(boostingTree):
    def get_base_model(self):
        from xgboost import XGBClassifier
        return XGBClassifier


class GBDT(boostingTree):
    def get_base_model(self):
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier


class CatBoost(boostingTree):
    def __init__(self, collaborate_feature_num=30, column_description=None, model_dir="./", **kwargs):
        super(CatBoost, self).__init__(collaborate_feature_num, **kwargs)
        from longling.base import string_types
        self.cd_file = self.generate_cd_file(column_description) if isinstance(column_description, string_types) else column_description
        self.model_dir = model_dir

    def get_base_model(self):
        from catboost import CatBoostClassifier
        return CatBoostClassifier

    def fit(self, **kwargs):
        model = self.base_model(**self.kwargs)
        model.fit(**kwargs)
        self.model = model

    def pool(self, datas, cd_file=None, **kwargs):
        if cd_file is None:
            cd_file = self.cd_file
        from catboost import Pool
        return Pool(datas, column_description=cd_file, **kwargs)

    def generate_cd_file(self, column_description, wf_path=None):
        if wf_path is None:
            from os import path
            wf_path = path.join(self.model_dir, "column_description.cd")

        if isinstance(column_description, dict):
            from longling.lib.stream import wf_open, wf_close
            wf = wf_open(wf_path)
            for i, name_value in enumerate(column_description.items()):
                name, value = name_value
                if value == "Categ":
                    value += (" " + name)
                print("%s %s" % (i, value), file=wf)
            wf_close(wf)
        else:
            raise TypeError()

        return wf_path


def split_train_val_test(train_file_path, train_ratio, val_ratio, test_ratio):
    '''
    split train data to train, val and test according to ratio
    :param train_file:
    :param val_ratio:
    :param test_ratio:
    :return:
    '''
    import random
    import codecs
    from tqdm import tqdm
    train_file = codecs.open(train_file_path, 'r', 'utf-8')
    train_add_random_file = codecs.open("./train_add_flag.csv", 'w', 'utf-8')
    train_out_file = codecs.open("./train.csv", 'w', 'utf-8')
    validation_out_file = codecs.open("./validation.csv", 'w', 'utf-8')
    test_out_file = codecs.open("./test.csv", 'w', 'utf-8')

    title = train_file.readline()
    train_add_random_file.writelines(title.strip() + ",random\n")
    train_out_file.writelines(title)
    validation_out_file.writelines(title)
    test_out_file.writelines(title)

    count = 0
    train_count = 0
    val_count = 0
    test_count = 0

    for line in tqdm(train_file):
        flag = random.random()
        assert len(line.strip().split(',')) == len(title.split(',')), "\n" + unicode(
            len(title.split(','))) + '|' + title + unicode(len(line.strip().split(','))) + '|' + line
        train_add_random_file.write("{0},{1}\n".format(line.strip(), flag))
        if flag < train_ratio:
            train_out_file.write(line)
            train_count += 1
        elif flag < train_ratio + val_ratio:
            validation_out_file.write(line)
            val_count += 1
        else:
            test_out_file.write(line)
            test_count += 1
        count += 1
        if count % 100000 == 0:
            print("split data: {0}".format(count))

    print("new train data size: {0}".format(train_count))
    print("new validation data size: {0}".format(val_count))
    print("new test data size: {0}".format(test_count))

    train_file.close()
    train_add_random_file.close()
    validation_out_file.close()
    test_out_file.close()
    train_out_file.close()


def train_val_catboost(train_file_path, val_file_path, cd_file_path, param={'iterations': 100, 'learning_rate': 0.1}):
    # Load data from files to Pool
    train_pool = Pool(train_file_path, column_description=cd_file_path, delimiter=',')
    valid_pool = Pool(val_file_path, column_description=cd_file_path, delimiter=',')

    # Initialize CatBoostClassifier
    print("Start train model...")
    clf = CatBoostClassifier(param)
    clf.fit(train_pool, use_best_model=True, eval_set=valid_pool)
    # clf.fit(train_pool)

    # Save model
    print("Start save model...")
    clf.save_model("catboost.model", format="cbm", export_parameters=None)


def train_test_catboost(train_file_path, test_file_path, cd_file_path, submission_file_path,
                        param_str="iterations=100, learning_rate=0.1"):
    # Load data from files to Pool
    print("Start load data...")
    train_pool = Pool(train_file_path, column_description=cd_file_path, delimiter=',', has_header=True)
    test_pool = Pool(test_file_path, column_description=cd_file_path, delimiter=',', has_header=True)

    # Initialize CatBoostClassifier
    print("Start train model...")
    clf = eval("CatBoostClassifier({0})".format(param_str))
    clf.fit(train_pool)

    # Save model
    print("Start save model...")
    clf.save_model("catboost.model", format="cbm", export_parameters=None)

    # Predict
    print("Start predict...")
    result = clf.predict_proba(test_pool)
    print("Predict finished, start write submission file...")
    submission_file = open(submission_file_path, 'w')
    submission_file.writelines("id,target\n")
    id = 0
    for r in result:
        submission_file.writelines("{0},{1}\n".format(id, r))
        id += 1
        if id % 100000 == 0:
            print("predict count:" + str(id))


def predict(model_path, predict_pool):
    pass


def generate_cd_file():
    # Generate cd file
    train = open("train.csv", 'r')
    test = open("test.csv", 'r')
    train_title = train.readline()
    test_title = test.readline()
    train_title_list = train_title.strip().split(',')
    test_title_list = test_title.strip().split(',')

    count = 0
    for t in train_title_list:
        print("{0}\tCateg\t{1}".format(count, t))
        count += 1
    print("\n")

    count = 0
    for t in test_title_list:
        print("{0}\tCateg\t{1}".format(count, t))
        count += 1


if __name__ == '__main__':

    TRAIN_FILE = "train.csv"
    # VALID_FILE = "validation_remove_title.csv"
    TEST_FILE = "test.csv"
    CD_FILE = "column_description_37.cd"
    SUBMISSION_FILE = "submission_37.csv"

    train_test_catboost(train_file_path=TRAIN_FILE,
                        test_file_path=TEST_FILE,
                        cd_file_path=CD_FILE,
                        submission_file_path=SUBMISSION_FILE)

    # # Load data from files to Pool
    # train_pool = Pool(TRAIN_FILE, column_description=CD_FILE, delimiter=',')
    # valid_pool = Pool(VALID_FILE, column_description=CD_FILE, delimiter=',')
    # test_pool = Pool(TEST_FILE, column_description=CD_FILE, delimiter=',', has_header=True)
    #
    # clf = CatBoostClassifier.load_model("catboost.model")
    # result = clf.predict_proba(test_pool)
    # submission_file = codecs.open(SUBMISSION_FILE, 'w', 'utf-8')
    # submission_file.writelines("id,target\n")
    # id = 0
    # for r in result:
    #     submission_file.writelines("{0},{1}\n".format(id, r))
    #     id += 1
    #     if id % 100000:
    #         print(id)

    # infile = open("submission.csv", 'r')
    # outfile = open("submission_final.csv", 'w')
    # outfile.writelines(infile.readline())
    # id = 0
    # for line in infile:
    #     outfile.writelines(str(id) + "," + line.strip().split(' ')[-1][:-1] + "\n")
    #     id += 1

