# coding: utf-8
# 2021/3/16 @ tongshiwei
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.nn.functional as F
from longling.ML.metrics import classification_report

from longling.ML.PytorchHelper.utils import Configuration
from longling.ML.PytorchHelper import fit_wrapper, loss_dict2tmt_torch_loss, eval_wrapper


def transform(x, y, batch_size, **params):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.int64)
    )
    return DataLoader(dataset, batch_size=batch_size, **params)


def etl(batch_size):
    # extract
    digits = datasets.load_digits()

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # split dataset into train, validation and test
    # we set (train, validation): test = 8 : 2 and train : validation = 9 : 1
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.2, shuffle=False)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.1, shuffle=False)

    return transform(X_train, y_train, batch_size), transform(X_valid, y_valid, batch_size)


class MLP(nn.Module):
    def __init__(self, in_feats, hidden_layers=None, out=10, **kwargs):
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


def config():
    configuration = Configuration(model_name="mlp", model_dir="mlp")
    configuration.end_epoch = 2
    configuration.batch_size = 32
    configuration.hyper_params = {"hidden_layers": [512, 128]}

    return configuration


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


def get_net(*args, **kwargs):
    return MLP(in_feats=64, *args, **kwargs)


def get_loss(*args, **kwargs):
    return loss_dict2tmt_torch_loss(
        {"cross entropy": torch.nn.CrossEntropyLoss(*args, **kwargs)}
    )


if __name__ == '__main__':
    from longling.ML.PytorchHelper import light_module as lm

    cfg = config()
    train_data, valid_data = etl(cfg.batch_size)

    cfg.dump()

    lm.train(
        net=None,
        cfg=cfg,
        loss_function=None,
        get_loss=get_loss,
        trainer=None,
        train_data=train_data,
        test_data=valid_data,
        fit_f=fit_f,
        eval_f=eval_f,
        initial_net=True,
        get_net=get_net,
        params_save=True,  # enable parameter persistence
        dump_result=True
    )
