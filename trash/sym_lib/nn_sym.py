# coding: utf-8
# create by tongshiwei on 2018/7/7
"""
此文件定义了一些新的symbol
"""

import mxnet as mx

from longling.lib.candylib import Register


@mx.init.register
class MyConstant(mx.init.Initializer):
    def __init__(self, value):
        super(MyConstant, self).__init__(value=value)
        self.value = value

    def _init_weight(self, _, arr):
        arr[:] = mx.nd.array(self.value)


class MXConstantRegister(Register):
    def __init__(self):
        super(MXConstantRegister, self).__init__()
        self.idx = 0
        self.default_name = "{}Constant{}"

    def get_name(self, name):
        name = self.default_name.format(name, self.idx)
        self.idx += 1
        return name


mx_constant_register = MXConstantRegister()


def mx_constant(value, name=None, attr=None, shape=None, dtype=None, init=MyConstant, stype=None, **kwargs):
    """
    Creates a symbolic variable with specified name.

    Example
    -------
    >>> data = mx.sym.Variable('data', attr={'a': 'b'})
    >>> data
    <Symbol data>
    >>> csr_data = mx.sym.Variable('csr_data', stype='csr')
    >>> csr_data
    <Symbol csr_data>
    >>> row_sparse_weight = mx.sym.Variable('weight', stype='row_sparse')
    >>> row_sparse_weight
    <Symbol weight>

    Parameters
    ----------
    value: int or float or list
    name : str
        Variable name.
    attr : Dict of strings
        Additional attributes to set on the variable. Format {string : string}.
    shape : tuple
        The shape of a variable. If specified, this will be used during the shape inference.
        If one has specified a different shape for this variable using
        a keyword argument when calling shape inference, this shape information will be ignored.
    dtype : str or numpy.dtype
        The dtype for input variable. If not specified, this value will be inferred.
    init : initializer (mxnet.init.*)
        Initializer for this variable to (optionally) override the default initializer.
    stype : str
        The storage type of the variable, such as 'row_sparse', 'csr', 'default', etc
    kwargs : Additional attribute variables
        Additional attributes must start and end with double underscores.

    Returns
    -------
    variable : Symbol
        A symbol corresponding to an input to the computation graph.
    """
    name = "" if name is None else name
    name = mx_constant_register.get_name(name)
    mx_constant_register.register(name)

    return mx.sym.Variable(
        name=name,
        attr=attr,
        shape=shape,
        lr_mult=0,
        wd_mult=0,
        dtype=dtype,
        init=init(value),
        stype=stype,
        **kwargs
    )


def pairwise_loss(pos_sym, neg_sym, margin):
    margin = mx_constant([margin])
    loss = mx.sym.add_n(mx.sym.negative(pos_sym), neg_sym, margin)
    sym = mx.sym.relu(loss)
    loss = mx.sym.MakeLoss(sym)
    return loss
