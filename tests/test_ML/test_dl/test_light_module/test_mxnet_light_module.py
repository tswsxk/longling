# coding: utf-8
# 2021/8/4 @ tongshiwei
from longling.ML.MxnetHelper import light_module as lm
import pytest
import mxnet as mx
from mxnet import gluon
from mxnet import nd
from mxnet.gluon.data import ArrayDataset, DataLoader
from longling.ML.metrics import classification_report

from longling.ML.MxnetHelper import Configuration
from longling.ML.MxnetHelper import fit_wrapper, loss_dict2tmt_mx_loss


def etl(data_x, data_y, cfg: Configuration):
    batch_size = cfg.batch_size
    dataset = ArrayDataset(
        mx.nd.array(data_x),
        mx.nd.array(data_y)
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


def eval_f(_net, test_data, ctx=mx.cpu()):
    y_true = []
    y_pred = []
    for x, y in test_data:
        x = x.as_in_context(ctx)
        pred = _net(x).argmax(-1).asnumpy().tolist()
        y_pred.extend(pred)
        y_true.extend(y.asnumpy().tolist())

    return classification_report(y_true, y_pred)


class MLP(gluon.HybridBlock):
    def __init__(self, hidden_layers=None, out=2, **kwargs):
        super(MLP, self).__init__()
        with self.name_scope():
            self.nn = gluon.nn.HybridSequential()
            if hidden_layers is not None:
                for hidden_unit in hidden_layers:
                    self.nn.add(
                        gluon.nn.Dense(hidden_unit, activation="tanh"),
                        gluon.nn.Dropout(0.5)
                    )
            self.nn.add(
                gluon.nn.Dense(out)
            )

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.log_softmax(self.nn(x))


def get_net(in_feats, *args, **kwargs):
    return MLP(in_feats=in_feats, *args, **kwargs)


def get_loss(*args, **kwargs):
    return loss_dict2tmt_mx_loss(
        {"cross entropy": gluon.loss.SoftmaxCELoss(*args, **kwargs)}
    )


@pytest.mark.filterwarnings("ignore:Precision and F-score")
def test_mxnet_light_module(data_x, data_y, constants, tmpdir):
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
        return nd.softmax_cross_entropy(x, y)

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
    configuration.lr_params = {"scheduler": "linear", "learning_rate": 0.01, "max_update": 20}

    # when lr_params is set, batch lr scheduler or epoch lr scheduler is expected to be specified
    lm.train(
        net=None,
        cfg=configuration,
        loss_function=lambda x, y: nd.softmax_cross_entropy(x, y),
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
        loss_function=lambda x, y: nd.softmax_cross_entropy(x, y),
        trainer=None,
        train_data=batch_data,
        test_data=batch_data,
        fit_f=fit_f,
        eval_f=eval_f,
        initial_net=True,
        get_net=get_net,
        loss_as_dict=True,
    )

    # use loss class and use tqdm as progress monitor and epoch lr scheduler
    configuration.lr_params = {"update_params": {"scheduler": "linear", "learning_rate": 0.01}}
    loss = gluon.loss.SoftmaxCELoss()
    net = get_net(**configuration.hyper_params)
    net.initialize()
    lm.train(
        net=net,
        loss_function=loss,
        cfg=configuration,
        trainer=None,
        train_data=batch_data,
        test_data=batch_data,
        fit_f=fit_f,
        eval_f=eval_f,
        initial_net=False,
        loss_as_dict=True,
        progress_monitor="tqdm",
        dump_result=True,
        params_save=True,
    )
