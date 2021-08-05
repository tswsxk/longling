# coding: utf-8
# 2021/8/4 @ tongshiwei

import pytest
import mxnet as mx
from mxnet.gluon import nn
from longling.ML.MxnetHelper import save_params, load_net, set_embedding_weight, array_equal
from longling.ML.DL import ALL_PARAMS, BLOCK_EMBEDDING


class EmbedMLP(nn.HybridBlock):
    def __init__(self):
        super(EmbedMLP, self).__init__()
        self.embedding = nn.Embedding(5, 3)
        self.linear = nn.Dense(5)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.linear(self.embedding(x))


def test_save_load(tmpdir):
    model_dir = tmpdir.mkdir("torch")
    module = EmbedMLP()
    module.initialize()
    set_embedding_weight(module.embedding, mx.nd.ones((5, 3)))

    data = mx.nd.array([0, 1, 2])

    vec1 = module(data)

    model_params = save_params(str(model_dir.join("mlp.params")), module, select=ALL_PARAMS)
    model_params_no_embedding = save_params(str(model_dir.join("mlp_noe.params")), module, select=BLOCK_EMBEDDING)
    module2 = EmbedMLP()
    load_net(model_params, module2)
    vec2 = module2(data)

    assert array_equal(vec1, vec2)

    module3 = EmbedMLP()
    module3.initialize()

    set_embedding_weight(module3.embedding, mx.nd.zeros((5, 3)))
    load_net(model_params_no_embedding, module3, allow_missing=True)
    vec3 = module3(data)
    assert array_equal(vec1, vec3) is False

    with pytest.raises(FileExistsError):
        load_net("error", module3)
