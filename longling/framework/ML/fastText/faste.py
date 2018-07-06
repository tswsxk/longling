# coding:utf-8
# created by tongshiwei on 2018/7/6

'''
此文件用来展示fastText模块使用方法
'''

from longling.framework.ML.fastText.io_lib import jsonxz2fast
from longling.framework.ML.fastText.Fasttext import Fasttext

from longling.framework.ML.universe.fileio.jsonxz import load_jsonxz

from sklearn.metrics import classification_report


if __name__ == '__main__':
    root = "../../../data/fasttext/"
    location_train = root + "train"
    location_test = root + "test"

    model = Fasttext()
    model.fit(location_train, root + "model/", cast_file_func=jsonxz2fast)

    datas, labels = load_jsonxz(location_test)
    preds = model.predict(datas)

    classification_report(labels, preds)

