# pylint:skip-file
import sys
sys.path.insert(0, "../../python")
sys.path.insert(0,"/opt/tiger/nlp/text_env/env/lib/python2.7/site-packages/mxnet-0.9.5-py2.7.egg")
import mxnet as mx
from collections import namedtuple
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])


def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """LSTM Cell symbol"""
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_i2h" % (seqidx, layeridx))
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name="t%d_l%d_h2h" % (seqidx, layeridx))
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name="t%d_l%d_slice" % (seqidx, layeridx))
    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
    return LSTMState(c=next_c, h=next_h)


# we define a new unrolling function here because the original
# one in lstm.py concats all the labels at the last layer together,
# making the mini-batch size of the label different from the data.
# I think the existing data-parallelization code need some modification
# to allow this situation to work properly
def lstm_unroll(num_lstm_layer, seq_len, input_size,
                num_hidden, num_embed, num_label, dropout=0.):

    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable("l%d_i2h_weight" % i),
                                     i2h_bias=mx.sym.Variable("l%d_i2h_bias" % i),
                                     h2h_weight=mx.sym.Variable("l%d_h2h_weight" % i),
                                     h2h_bias=mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)

    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    #embed = mx.sym.Embedding(data=data, input_dim=input_size,
    #                         weight=embed_weight, output_dim=num_embed, name='embed')
    #wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)

    wordvec = mx.sym.SliceChannel(data=data, num_outputs=seq_len)
    # wordvec = mx.sym.Reshape(data=data, shape=(0, seq_len, -1))
    # wordvec = mx.sym.SliceChannel(data=wordvec, num_outputs=seq_len, axis=1, squeeze_axis=True)
    hidden_all = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]

        # stack LSTM
        for i in range(num_lstm_layer):
            if i == 0:
                dp_ratio = 0.
            else:
                dp_ratio = dropout
            next_state = lstm(num_hidden, indata=hidden,
                              prev_state=last_states[i],
                              param=param_cells[i],
                              seqidx=seqidx, layeridx=i, dropout=dp_ratio)
            hidden = next_state.h
            last_states[i] = next_state
        # decoder
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        hidden_all.append(hidden)

    # hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    # pred = mx.sym.FullyConnected(data=pool, num_hidden=num_label,
    #                              weight=cls_weight, bias=cls_bias, name='pred')

    # hidden_concat = mx.sym.Concat(*hidden_all, dim=0) # [seq_len * batch_size, dim]
    pred = mx.sym.FullyConnected(data=hidden_all[-1], num_hidden=num_label,
                                weight=cls_weight, bias=cls_bias, name='pred')

    ################################################################################
    # Make label the same shape as our produced data path
    # I did not observe big speed difference between the following two ways

    # label = mx.sym.transpose(data=label)
    label = mx.sym.Reshape(data=label, shape=(-1, ))

    #label_slice = mx.sym.SliceChannel(data=label, num_outputs=seq_len)
    #label = [label_slice[t] for t in range(seq_len)]
    #label = mx.sym.Concat(*label, dim=0)
    #label = mx.sym.Reshape(data=label, target_shape=(0,))
    ################################################################################

    sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    return sm


def lstm_inference_symbol(num_lstm_layer, input_size,
                          num_hidden, num_embed, num_label, dropout=0.):
    seqidx = 0
    embed_weight=mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    param_cells = []
    last_states = []
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight = mx.sym.Variable("l%d_i2h_weight" % i),
                                      i2h_bias = mx.sym.Variable("l%d_i2h_bias" % i),
                                      h2h_weight = mx.sym.Variable("l%d_h2h_weight" % i),
                                      h2h_bias = mx.sym.Variable("l%d_h2h_bias" % i)))
        state = LSTMState(c=mx.sym.Variable("l%d_init_c" % i),
                          h=mx.sym.Variable("l%d_init_h" % i))
        last_states.append(state)
    assert(len(last_states) == num_lstm_layer)
    data = mx.sym.Variable("data")

    #hidden = mx.sym.Embedding(data=data,
    #                          input_dim=input_size,
    #                          output_dim=num_embed,
    #                          weight=embed_weight,
    #                          name="embed")
    hidden = data#fix

    # stack LSTM
    for i in range(num_lstm_layer):
        if i==0:
            dp=0.
        else:
            dp = dropout
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[i],
                          param=param_cells[i],
                          seqidx=seqidx, layeridx=i, dropout=dp)
        hidden = next_state.h
        last_states[i] = next_state
    # decoder
    if dropout > 0.:
        hidden = mx.sym.Dropout(data=hidden, p=dropout)
    fc = mx.sym.FullyConnected(data=hidden, num_hidden=num_label,
                               weight=cls_weight, bias=cls_bias, name='pred')
    sm = mx.sym.SoftmaxOutput(data=fc, name='softmax')
    output = [sm]
    for state in last_states:
        output.append(state.c)
        output.append(state.h)
    output.append(hidden)
    return mx.sym.Group(output)


def bi_lstm_unroll(seq_len, input_size,
                   num_hidden, num_embed, num_label, dropout=0.):

    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    last_states = []
    last_states.append(LSTMState(c = mx.sym.Variable("l0_init_c"), h = mx.sym.Variable("l0_init_h")))
    last_states.append(LSTMState(c = mx.sym.Variable("l1_init_c"), h = mx.sym.Variable("l1_init_h")))
    forward_param = LSTMParam(i2h_weight=mx.sym.Variable("l0_i2h_weight"),
                              i2h_bias=mx.sym.Variable("l0_i2h_bias"),
                              h2h_weight=mx.sym.Variable("l0_h2h_weight"),
                              h2h_bias=mx.sym.Variable("l0_h2h_bias"))
    backward_param = LSTMParam(i2h_weight=mx.sym.Variable("l1_i2h_weight"),
                              i2h_bias=mx.sym.Variable("l1_i2h_bias"),
                              h2h_weight=mx.sym.Variable("l1_h2h_weight"),
                              h2h_bias=mx.sym.Variable("l1_h2h_bias"))

    # embeding layer
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')
    #embed = mx.sym.Embedding(data=data, input_dim=input_size,
    #                         weight=embed_weight, output_dim=num_embed, name='embed')
    #wordvec = mx.sym.SliceChannel(data=embed, num_outputs=seq_len, squeeze_axis=1)
    #wordvec = mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)#fix
    wordvec = mx.sym.SliceChannel(data=data, num_outputs=seq_len)#fix

    forward_hidden = []
    for seqidx in range(seq_len):
        hidden = wordvec[seqidx]
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[0],
                          param=forward_param,
                          seqidx=seqidx, layeridx=0, dropout=dropout)
        hidden = next_state.h
        last_states[0] = next_state
        forward_hidden.append(hidden)

    backward_hidden = []
    for seqidx in range(seq_len):
        k = seq_len - seqidx - 1
        hidden = wordvec[k]
        next_state = lstm(num_hidden, indata=hidden,
                          prev_state=last_states[1],
                          param=backward_param,
                          seqidx=k, layeridx=1,dropout=dropout)
        hidden = next_state.h
        last_states[1] = next_state
        backward_hidden.insert(0, hidden)

    hidden_all = []
    for i in range(seq_len):
        hidden_all.append(mx.sym.Concat(*[forward_hidden[i], backward_hidden[i]], dim=1))

    #hidden_concat = mx.sym.Concat(hidden_all[-1], dim=0)
    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    pred = mx.sym.FullyConnected(data=hidden_all[-1], num_hidden=num_label,
                                 weight=cls_weight, bias=cls_bias, name='pred')

    label = mx.sym.transpose(data=label)
    label = mx.sym.Reshape(data=label, shape=(-1,))
    sm = mx.sym.SoftmaxOutput(data=pred, label=label, name='softmax')

    return sm


def bi_lstm_inference_symbol(input_size, seq_len,
                             num_hidden, num_embed, num_label, dropout=0.):
    seqidx = 0
    embed_weight=mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")
    last_states = [LSTMState(c = mx.sym.Variable("l0_init_c"), h = mx.sym.Variable("l0_init_h")),
                   LSTMState(c = mx.sym.Variable("l1_init_c"), h = mx.sym.Variable("l1_init_h"))]
    forward_param = LSTMParam(i2h_weight=mx.sym.Variable("l0_i2h_weight"),
                              i2h_bias=mx.sym.Variable("l0_i2h_bias"),
                              h2h_weight=mx.sym.Variable("l0_h2h_weight"),
                              h2h_bias=mx.sym.Variable("l0_h2h_bias"))
    backward_param = LSTMParam(i2h_weight=mx.sym.Variable("l1_i2h_weight"),
                              i2h_bias=mx.sym.Variable("l1_i2h_bias"),
                              h2h_weight=mx.sym.Variable("l1_h2h_weight"),
                              h2h_bias=mx.sym.Variable("l1_h2h_bias"))
    data = mx.sym.Variable("data")
    #embed = mx.sym.Embedding(data=data, input_dim=input_size,
    #                         weight=embed_weight, output_dim=num_embed, name='embed')
    wordvec = mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)#fix
    forward_hidden = []
    for seqidx in range(seq_len):
        next_state = lstm(num_hidden, indata=wordvec[seqidx],
                          prev_state=last_states[0],
                          param=forward_param,
                          seqidx=seqidx, layeridx=0, dropout=0.0)
        hidden = next_state.h
        last_states[0] = next_state
        forward_hidden.append(hidden)

    backward_hidden = []
    for seqidx in range(seq_len):
        k = seq_len - seqidx - 1
        next_state = lstm(num_hidden, indata=wordvec[k],
                          prev_state=last_states[1],
                          param=backward_param,
                          seqidx=k, layeridx=1, dropout=0.0)
        hidden = next_state.h
        last_states[1] = next_state
        backward_hidden.insert(0, hidden)

    hidden_all = []
    for i in range(seq_len):
        hidden_all.append(mx.sym.Concat(*[forward_hidden[i], backward_hidden[i]], dim=1))
    hidden_concat = mx.sym.Concat(*hidden_all, dim=0)
    fc = mx.sym.FullyConnected(data=hidden_all[-1], num_hidden=num_label,
                               weight=cls_weight, bias=cls_bias, name='pred')
    sm = mx.sym.SoftmaxOutput(data=fc, name='softmax')
    output = [sm]
    for state in last_states:
        output.append(state.c)
        output.append(state.h)
    return mx.sym.Group(output)
