# coding:utf-8
# created by tongshiwei on 2018/8/2

from mxnet import symbol, ndarray
from mxnet.gluon.parameter import tensor_types

from longling.lib.candylib import as_list

__all__ = ["format_sequence", "mask_sequence_variable_length"]


def format_sequence(length, inputs, layout, merge, in_layout=None):
    """

    Parameters
    ----------
    length
    inputs
    layout
    merge
    in_layout

    Returns
    -------

    """
    assert inputs is not None, \
        "unroll(inputs=None) has been deprecated. " \
        "Please create input variables outside unroll."

    axis = layout.find('T')
    batch_axis = layout.find('N')
    batch_size = 0
    in_axis = in_layout.find('T') if in_layout is not None else axis
    if isinstance(inputs, symbol.Symbol):
        F = symbol
        if merge is False:
            assert len(inputs.list_outputs()) == 1, \
                "unroll doesn't allow grouped symbol as input. " \
                "Please convert " \
                "to list with list(inputs) first or " \
                "let unroll handle splitting."
            inputs = list(symbol.split(
                inputs, axis=in_axis, num_outputs=length, squeeze_axis=1
            ))
    elif isinstance(inputs, ndarray.NDArray):
        F = ndarray
        batch_size = inputs.shape[batch_axis]
        if merge is False:
            assert length is None or length == inputs.shape[in_axis]
            inputs = as_list(ndarray.split(inputs, axis=in_axis,
                                           num_outputs=inputs.shape[in_axis],
                                           squeeze_axis=1))
    else:
        assert length is None or len(inputs) == length
        if isinstance(inputs[0], symbol.Symbol):
            F = symbol
        else:
            F = ndarray
            batch_size = inputs[0].shape[batch_axis]
        if merge is True:
            inputs = F.stack(*inputs, axis=axis)
            in_axis = axis

    if isinstance(inputs, tensor_types) and axis != in_axis:
        inputs = F.swapaxes(inputs, dim1=axis, dim2=in_axis)

    return inputs, axis, F, batch_size


def mask_sequence_variable_length(F, data, length, valid_length, time_axis,
                                  merge):
    """

    Parameters
    ----------
    F
    data
    length
    valid_length
    time_axis
    merge

    Returns
    -------

    """
    assert valid_length is not None
    if not isinstance(data, tensor_types):
        data = F.stack(*data, axis=time_axis)
    outputs = F.SequenceMask(
        data, sequence_length=valid_length,
        use_sequence_length=True,
        axis=time_axis
    )
    if not merge:
        outputs = as_list(
            F.split(
                outputs, num_outputs=length, axis=time_axis, squeeze_axis=True
            )
        )
    return outputs
