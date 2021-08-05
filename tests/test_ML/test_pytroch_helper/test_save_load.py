# coding: utf-8
# 2021/8/4 @ tongshiwei

import pytest
import torch
from torch import nn
from longling.ML.PytorchHelper import save_params, load_net
from longling.ML.DL import ALL_PARAMS, BLOCK_EMBEDDING


class EmbedMLP(nn.Module):
    def __init__(self):
        super(EmbedMLP, self).__init__()
        self.embedding = nn.Embedding(5, 3)
        self.linear = nn.Linear(3, 5)

    def forward(self, x):
        return self.linear(self.embedding(x))


def test_save_load(tmpdir):
    model_dir = tmpdir.mkdir("torch")
    module = EmbedMLP()
    module.embedding.from_pretrained(torch.ones(5, 3))

    data = torch.tensor([0, 1, 2])

    vec1 = module(data)

    model_params = save_params(str(model_dir.join("mlp.params")), module, select=ALL_PARAMS)
    model_params_no_embedding = save_params(str(model_dir.join("mlp_noe.params")), module, select=BLOCK_EMBEDDING)
    module2 = EmbedMLP()
    load_net(model_params, module2)
    vec2 = module2(data)

    assert torch.equal(vec1, vec2)

    module3 = EmbedMLP()

    module3.embedding.from_pretrained(torch.zeros(5, 3))
    load_net(model_params_no_embedding, module3, allow_missing=True)
    vec3 = module3(data)
    assert torch.equal(vec1, vec3) is False

    with pytest.raises(FileExistsError):
        load_net("error", module3)
