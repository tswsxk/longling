# coding:utf-8
import sys

sys.path.insert(0, "/opt/tiger/text_lib/env/lib/python2.7/site-packages")
import fasttext

LABEL_PREFIX = '__label__'


class TextFasttextScorer(object):
    def __init__(self, params):
        location_model = params['location_model']
        self.classifier = fasttext.load_model(location_model, label_prefix=LABEL_PREFIX)

    @staticmethod
    def check_parameters(params):
        return 'location_model' in params

    def get_score(self, sentence):
        if isinstance(sentence, str):
            sentence = sentence.decode('utf8')
        xx = sentence.split()
        if len(xx) == 0:
            return 0.
        texts = [' '.join(xx)]
        res = self.classifier.predict_proba(texts, 1)
        label, prob = res[0][0]
        if label == '0':
            prob = 1. - prob
        return prob


if __name__ == '__main__':
    print 'x'

    location_model = '../../data/model/vulgar/model.bin'

    sentence = u'骚气'
    scorer = TextFasttextScorer({
        'location_model': location_model,
    })

    res = scorer.get_score(sentence)
    print res
