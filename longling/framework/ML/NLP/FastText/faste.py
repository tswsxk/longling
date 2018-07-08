# coding:utf-8
# created by tongshiwei on 2018/7/6

'''
此文件用来展示fastText模块使用方法
'''

from sklearn.metrics import classification_report

from longling.framework.ML.NLP.FastText import FastText
from longling.framework.ML.NLP.FastText import jsonxz2fast
from longling.framework.ML.universe.fileio.jsonxz import load_jsonxz

if __name__ == '__main__':
    root = "../../../../../data/fasttext/TextSim/"
    location_train = root + "train"
    location_test = root + "test"

    model = FastText()
    model.fit(location_train, root + "model/", cast_file_func=jsonxz2fast, array_tag=False)

    datas, labels = load_jsonxz(location_test)
    preds = [data[0] for data in model.predict(datas)]

    print(classification_report(labels, preds, digits=3))

