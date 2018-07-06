# coding:utf-8
# created by tongshiwei on 2018/7/6

'''
此文件用来展示fastText模块使用方法
'''

from longling.framework.ML.fastText.io_lib import jsonxz2fast
from longling.framework.ML.fastText.fasttext import Fasttext

from longling.framework.ML.universe.fileio.jsonxz import load_jsonxz

from sklearn.metrics import classification_report


if __name__ == '__main__':
    root = "../../../../data/fasttext/"
    location_jsonxz = root + "train/text_test_train"
    jsonxz2fast(location_jsonxz)

    model = Fasttext()
    model.fit(location_jsonxz, root + "model/")

    datas, labels = load_jsonxz(location_jsonxz)
    preds = model.predict(datas)

    classification_report(labels, preds)

