# coding: utf-8
# 2021/8/4 @ tongshiwei

import pytest
import torch
from torch import nn
import torch.nn.functional as F
from longling.ML.PytorchHelper import light_module as lm
from longling.ML.PytorchHelper import fit_wrapper, loss_dict2tmt_torch_loss, eval_wrapper
from longling.ML.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader
from longling.ML.PytorchHelper.utils import Configuration


def etl(data_x, data_y, cfg: Configuration):
    batch_size = cfg.batch_size
    dataset = TensorDataset(
        torch.tensor(data_x),
        torch.tensor(data_y)
    )
    return DataLoader(dataset, batch_size=batch_size)


@fit_wrapper
def fit_f(_net, batch_data, loss_function, *args, **kwargs):
    x, y = batch_data
    out = _net(x)
    loss = []
    for _f in loss_function.values():
        loss.append(_f(out, y))
    return sum(loss)


@eval_wrapper
def eval_f(_net, test_data, *args, **kwargs):
    y_true = []
    y_pred = []
    for x, y in test_data:
        pred = _net(x).argmax(-1).tolist()
        y_pred.extend(pred)
        y_true.extend(y.tolist())
    return classification_report(y_true, y_pred)


class MLP(nn.Module):
    def __init__(self, in_feats, hidden_layers=None, out=2, **kwargs):
        super(MLP, self).__init__()
        layers = []
        if hidden_layers is not None:
            for i, units in enumerate([in_feats] + hidden_layers[:-1]):
                layers.append(nn.Linear(units, hidden_layers[i]))
                layers.append(nn.Dropout(0.5))
        layers.append(nn.Linear(hidden_layers[-1], out))
        self.nn = nn.Sequential(*layers)

    def forward(self, x):
        return F.log_softmax(self.nn(x), dim=-1)


def get_net(in_feats, *args, **kwargs):
    return MLP(in_feats=in_feats, *args, **kwargs)


def get_loss(*args, **kwargs):
    return loss_dict2tmt_torch_loss(
        {"cross entropy": torch.nn.CrossEntropyLoss(*args, **kwargs)}
    )


@pytest.mark.filterwarnings("ignore:Precision and F-score")
def test_torch_light_module(data_x, data_y, constants, tmpdir):
    in_feats = constants
    model_dir = str(tmpdir.mkdir("mlp"))
    configuration = Configuration(
        model_name="mlp",
        model_dir=model_dir,
        hyper_params_update={"in_feats": in_feats, "hidden_layers": [16, 32]},
        end_epoch=2,
        batch_size=32,
    )
    batch_data = etl(data_x, data_y, configuration)
    lm.train(
        net=None,
        cfg=configuration,
        loss_function=None,
        get_loss=get_loss,
        trainer=None,
        train_data=batch_data,
        test_data=batch_data,
        fit_f=fit_f,
        eval_f=eval_f,
        initial_net=True,
        get_net=get_net,
    )

    # test hyper-params search with nni and use loss function
    def cross_entropy(x, y):
        return F.cross_entropy(x, y)

    lm.train(
        net=None,
        cfg=configuration,
        loss_function=cross_entropy,
        trainer=None,
        train_data=batch_data,
        test_data=batch_data,
        fit_f=fit_f,
        eval_f=eval_f,
        initial_net=True,
        get_net=get_net,
        enable_hyper_search=True,
        primary_key="accuracy",
        loss_as_dict=True,
    )

    # test lambda loss function and batch lr scheduler
    configuration.lr_params = {"scheduler": "ExponentialLR", "gamma": 0.9}
    with pytest.raises(AssertionError):
        # when lr_params is set, batch lr scheduler or epoch lr scheduler is expected to be specified
        lm.train(
            net=None,
            cfg=configuration,
            loss_function=lambda x, y: F.cross_entropy(x, y),
            trainer=None,
            train_data=batch_data,
            test_data=batch_data,
            fit_f=fit_f,
            eval_f=eval_f,
            initial_net=True,
            get_net=get_net,
            loss_as_dict=True,
        )

    lm.train(
        net=None,
        cfg=configuration,
        loss_function=lambda x, y: F.cross_entropy(x, y),
        trainer=None,
        train_data=batch_data,
        test_data=batch_data,
        fit_f=fit_f,
        eval_f=eval_f,
        initial_net=True,
        get_net=get_net,
        loss_as_dict=True,
        batch_lr_scheduler=True
    )

    # default initialization, use loss class and use tqdm as progress monitor and epoch lr scheduler
    configuration.lr_params = {"scheduler": "OneCycleLR", "max_lr": 1,
                               "total_steps": configuration.end_epoch - configuration.begin_epoch}
    loss = torch.nn.CrossEntropyLoss()
    lm.train(
        net=None,
        loss_function=loss,
        cfg=configuration,
        trainer=None,
        train_data=batch_data,
        test_data=batch_data,
        fit_f=fit_f,
        eval_f=eval_f,
        initial_net=False,
        get_net=get_net,
        loss_as_dict=True,
        progress_monitor="tqdm",
        dump_result=True,
        params_save=True,
        epoch_lr_scheduler=True
    )
