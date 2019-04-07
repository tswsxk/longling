# coding: utf-8
# created by tongshiwei on 18-1-27

import traceback

import mxnet as mx

from longling.lib.stream import build_dir

__all__ = ["VizError", "plot_network"]


class VizError(Exception):
    pass


def plot_network(nn_symbol, save_path="plot/network", shape=None,
                 node_attrs=None, view=False):
    node_attrs = {} if node_attrs is None else node_attrs

    graph = mx.viz.plot_network(nn_symbol, shape=shape, node_attrs=node_attrs)

    assert save_path
    build_dir(save_path)
    try:
        graph.render(save_path, view=view)
    except Exception:
        raise VizError(traceback.format_exc())
