# coding:utf-8

from module_train import fit
import conf


class Params():
    def __init__(self, train_path, test_path, location_vec, model_dir, epoch, prop):
        self.gpus = prop.get('gpus', conf.GPU_DEFAULT)
        self.batch_size = prop.get('batch_size', conf.TRAIN_BATCH_SIZE)
        self.num_hidden = prop.get('num_hidden', conf.NUM_HIDDEN)
        # self.num_embed = prop.get('num_embed', conf.NUM_EMBED)
        self.buckets = prop.get("buckets", [])
        self.epoch_nums = epoch
        self.lr = prop.get('lr', conf.LR)
        self.momentum = prop.get('momentum', conf.MOMENTUM)
        self.num_label = prop.get('num_label', conf.NUM_LABEL)
        self.network = prop.get('network', 'lstm')
        self.num_lstm_layer = prop.get('num_lstm_layer', conf.NUM_LSTM_LAYER)
        self.train_path = train_path
        self.test_path = test_path
        self.location_vec = location_vec

        self.model_prefix = model_dir + '/' + "rnn"
        self.location_size = model_dir + '/' + "size.txt"


def process_fit(location_vec, location_ins, location_test, model_dir, gpu, prop={}, epoch=20):
    prop['gpus'] = prop.get('gpu', gpu)
    args = Params(location_ins, location_test, location_vec, model_dir, epoch, prop)
    fit(args)
