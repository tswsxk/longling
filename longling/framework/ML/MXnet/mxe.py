# coding: utf-8
# created by tongshiwei on 18-1-27


from __future__ import absolute_import

import json
import logging

import mxnet as mx
import numpy as np

from tqdm import tqdm

from longling.framework.ML.MXnet.callback import tqdm_speedometer, \
    ClassificationLogValidationMetricsCallback, Speedometer, \
    TqdmEpochReset
from longling.framework.ML.MXnet.metric import PRF, Accuracy, CrossEntropy
from longling.framework.ML.MXnet.viz import plot_network, form_shape
from longling.framework.ML.MXnet.io_lib import DictJsonIter, VecDict, SimpleBucketIter
from longling.framework.ML.MXnet.monitor import TimeMonitor
from longling.framework.ML.MXnet.sym_lib import lenet_cell, text_cnn_cell
from longling.framework.ML.MXnet.util import get_fine_tune_model

from longling.lib.utilog import config_logging
from longling.framework.spider import url_download


# todo
# plot data on website
# implement Jin experiment(r-cnn)
# GAN
# poem rnn
# picture to poem
# rcnn review
# rl network
# mxnet imgae iter

if __name__ == '__main__':
    pass
