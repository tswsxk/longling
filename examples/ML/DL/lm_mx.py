# coding: utf-8
# 2021/3/15 @ tongshiwei
import mxnet as mx
from mxnet import gluon
from sklearn import datasets
from sklearn.model_selection import train_test_split
from mxnet.gluon.data import ArrayDataset, DataLoader
from longling.ML.metrics import classification_report

from longling.ML.MxnetHelper.utils import Configuration


def transform(x, y, batch_size, **params):
    dataset = ArrayDataset(x.astype("float32"), y.astype("float32"))
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


class MLP(gluon.HybridBlock):
    def __init__(self, hidden_layers=None, out=10, **kwargs):
        super(MLP, self).__init__(**kwargs)
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


def config():
    configuration = Configuration(model_name="mlp", model_dir="mlp")
    configuration.end_epoch = 2
    configuration.batch_size = 32
    configuration.hyper_params = {"hidden_layers": [512, 128]}

    return configuration


def eval_f(_net, test_data, ctx=mx.cpu()):
    y_true = []
    y_pred = []
    for x, y in test_data:
        x = x.as_in_context(ctx)
        pred = _net(x).argmax(-1).asnumpy().tolist()
        y_pred.extend(pred)
        y_true.extend(y.asnumpy().tolist())

    return classification_report(y_true, y_pred)


if __name__ == '__main__':
    from longling.ML.MxnetHelper import light_module as lm
    from longling.ML.MxnetHelper import fit_wrapper, loss_dict2tmt_mx_loss

    cfg = config()
    train_data, valid_data = etl(cfg.batch_size)


    @fit_wrapper
    def fit_f(_net, batch_data, loss_function, *args, **kwargs):
        x, y = batch_data
        out = _net(x)
        loss = []
        for _value in loss_function.values():
            loss.append(_value(out, y))
        return sum(loss)


    def get_net(*args, **kwargs):
        return MLP(*args, **kwargs)


    def get_loss(*args, **kwargs):
        return loss_dict2tmt_mx_loss(
            {"cross entropy": gluon.loss.SoftmaxCELoss(*args, **kwargs)}
        )


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
    )
