# coding: utf-8
# 2021/8/5 @ tongshiwei

from mxnet import symbol, ndarray
from mxnet.gluon.parameter import tensor_types

from longling.lib.candylib import as_list

__all__ = ["format_sequence", "mask_sequence_variable_length", "get_begin_state"]


def format_sequence(length, inputs, layout, merge, in_layout=None):
    """
    `Original Code <https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/rnn/rnn_cell.py#L52>`_

    Parameters
    ----------
    length
    inputs
    layout
    merge
    in_layout

    Returns
    -------

    Examples
    --------
    >>> import mxnet.ndarray as nd
    >>> seq = [[[0] * 4, [2] * 4, [4] * 4], [[1] * 4, [3] * 4, [5] * 4]]
    >>> seq1, axis, _, batch_size = format_sequence(3, nd.array(seq), "NTC", False)
    >>> seq1   # doctest: +NORMALIZE_WHITESPACE
    [
    [[0. 0. 0. 0.]
     [1. 1. 1. 1.]]
    <NDArray 2x4 @cpu(0)>,
    [[2. 2. 2. 2.]
     [3. 3. 3. 3.]]
    <NDArray 2x4 @cpu(0)>,
    [[4. 4. 4. 4.]
     [5. 5. 5. 5.]]
    <NDArray 2x4 @cpu(0)>]
    >>> axis
    1
    >>> batch_size
    2
    >>> seq2, _, _, _ = format_sequence(3, nd.array(seq), "NTC", True)
    >>> seq2   # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    [[[0. 0. 0. 0.]
      [2. 2. 2. 2.]
      [4. 4. 4. 4.]]
    <BLANKLINE>
     [[1. 1. 1. 1.]
      [3. 3. 3. 3.]
      [5. 5. 5. 5.]]]
    <NDArray 2x3x4 @cpu(0)>
    >>> import mxnet.symbol as sym
    >>> seq3, _, _, _ = format_sequence(3, sym.Variable("s", shape=(2, 3, 4)), "NTC", False)
    >>> seq3
    [<Symbol split0>, <Symbol split0>, <Symbol split0>]
    >>> seq4 = [nd.array([[0] * 4, [1] * 4]), nd.array([[2] * 4, [3] * 4]), nd.array([[4] * 4, [5] * 4])]
    >>> seq5, _, _, _ = format_sequence(3, seq4, "NTC", True)
    >>> seq5   # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    [[[0. 0. 0. 0.]
      [2. 2. 2. 2.]
      [4. 4. 4. 4.]]
    <BLANKLINE>
     [[1. 1. 1. 1.]
      [3. 3. 3. 3.]
      [5. 5. 5. 5.]]]
    <NDArray 2x3x4 @cpu(0)>
    >>> seq6 = [sym.Variable("1", shape=(2, 4)), sym.Variable("2", shape=(2, 4)), sym.Variable("3", shape=(2, 4))]
    >>> seq7, _, _, _ = format_sequence(3, seq6, "NTC", True)
    >>> seq7
    <Symbol stack0>
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

    if isinstance(inputs, tensor_types) and axis != in_axis:  # pragma: no cover
        # todo: find the test case
        inputs = F.swapaxes(inputs, dim1=axis, dim2=in_axis)

    return inputs, axis, F, batch_size


def mask_sequence_variable_length(F, data, length, valid_length, time_axis,
                                  merge):
    """
    `Original Code <https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/rnn/rnn_cell.py#L82>`_

    Parameters
    ----------
    F
    data
    length
    valid_length
    time_axis
    merge: bool

    Returns
    -------
    masked_sequence: list, ...
        if merge is False, return list of step vector

    Examples
    --------
    >>> import mxnet.ndarray as nd
    >>> import mxnet as mx
    >>> mask_sequence_variable_length(
    ...     nd, mx.nd.ones((2, 4, 3)), 4, nd.array([2, 4]), 1, False
    ... )    # doctest: +NORMALIZE_WHITESPACE
    [
    [[1. 1. 1.]
     [1. 1. 1.]]
    <NDArray 2x3 @cpu(0)>,
    [[1. 1. 1.]
     [1. 1. 1.]]
    <NDArray 2x3 @cpu(0)>,
    [[0. 0. 0.]
     [1. 1. 1.]]
    <NDArray 2x3 @cpu(0)>,
    [[0. 0. 0.]
     [1. 1. 1.]]
    <NDArray 2x3 @cpu(0)>]
    >>> mask_sequence_variable_length(
    ...     nd, mx.nd.ones((2, 4, 3)), 4, nd.array([2, 4]), 1, True
    ... )    # doctest: +NORMALIZE_WHITESPACE
    <BLANKLINE>
    [[[1. 1. 1.]
      [1. 1. 1.]
      [0. 0. 0.]
      [0. 0. 0.]]
    <BLANKLINE>
     [[1. 1. 1.]
      [1. 1. 1.]
      [1. 1. 1.]
      [1. 1. 1.]]]
    <NDArray 2x4x3 @cpu(0)>
    >>> mask_sequence_variable_length(
    ...     nd, [mx.nd.ones((2, 3)), mx.nd.ones((2, 3)), mx.nd.ones((2, 3)), mx.nd.ones((2, 3))],
    ...     4, nd.array([2, 4]), 1, True
    ... )
    <BLANKLINE>
    [[[1. 1. 1.]
      [1. 1. 1.]
      [0. 0. 0.]
      [0. 0. 0.]]
    <BLANKLINE>
     [[1. 1. 1.]
      [1. 1. 1.]
      [1. 1. 1.]
      [1. 1. 1.]]]
    <NDArray 2x4x3 @cpu(0)>
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


def get_begin_state(cell, F, begin_state, inputs, batch_size):
    """
    `Original Code <https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/rnn/rnn_cell.py#L45>`_

    Parameters
    ----------
    cell
    F
    begin_state
    inputs
    batch_size

    Returns
    -------
    >>> import mxnet.ndarray as nd
    >>> from mxnet import gluon
    >>> lstm_cell = gluon.rnn.LSTMCell(3)
    >>> get_begin_state(lstm_cell, nd, None, nd.ones((2, 4, 5)), 2)   # doctest: +NORMALIZE_WHITESPACE
    [
    [[0. 0. 0.]
     [0. 0. 0.]]
    <NDArray 2x3 @cpu(0)>,
    [[0. 0. 0.]
     [0. 0. 0.]]
    <NDArray 2x3 @cpu(0)>]
    >>> import mxnet.symbol as sym
    >>> get_begin_state(lstm_cell, sym, None,
    ...     sym.Variable("inputs", shape=(2, 4, 5)), 2)   # doctest: +NORMALIZE_WHITESPACE
    [<Symbol lstm0_begin_state_2>, <Symbol lstm0_begin_state_3>]
    """
    if begin_state is None:
        if F is ndarray:
            ctx = inputs.context if isinstance(inputs, tensor_types) else inputs[0].context
            with ctx:
                begin_state = cell.begin_state(func=F.zeros, batch_size=batch_size)
        else:
            begin_state = cell.begin_state(func=F.zeros, batch_size=batch_size)
    return begin_state
