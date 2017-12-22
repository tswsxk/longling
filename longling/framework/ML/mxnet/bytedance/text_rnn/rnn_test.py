# -*- coding: utf-8 -*-
from predict import LstmPredict, BiLstmPredict

# lstm model
params = {
    "num_hidden": 128,
    "num_embed": 60,
    "num_label": 2,
    "num_lstm_layer": 1,
    "location_vec": 'data/title4.vec.dat',

    "model_prefix": 'model/m1',
    'epoch_num': 94,
    'idx_gpu': 0,
}
lstm_model = LstmPredict(params)

# bilstm model
params = {
    "num_hidden": 128,
    "num_embed": 60,
    "num_label": 2,
    "num_lstm_layer": 2,
    "location_vec": 'data/title4.vec.dat',

    "model_prefix": 'model/m2',
    'epoch_num': 94,
    'idx_gpu': 0,
}
bilstm_model = BiLstmPredict(params)


def test():
    title = u'爆乳 性感 少妇'
    p = lstm_model.process(title.split())
    print title, p

    title = u'爆乳 性感 少妇'
    p = bilstm_model.process(title.split())
    print title, p


if __name__ == '__main__':
    test()
