# coding: utf-8
# created by tongshiwei on 18-1-27

import mxnet as mx

from longling.lib.stream import build_dir

import traceback

class VizError(Exception):
    pass


def form_shape(data_iter):
    return {d[0]: d.shape for d in data_iter.provide_data}


def plot_network(nn_symbol, save_path="plot/network", shape=None, node_attrs={}, view=False):
    graph = mx.viz.plot_network(nn_symbol, shape=shape, node_attrs=node_attrs)

    assert save_path
    build_dir(save_path)
    try:
        graph.render(save_path, view=view)
    except Exception:
        raise VizError(traceback.format_exc())
