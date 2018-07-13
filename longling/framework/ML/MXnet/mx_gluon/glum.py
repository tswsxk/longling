# coding:utf-8
# created by tongshiwei on 2018/7/13

import mxnet as mx
from mxnet import autograd, nd
from mxnet import gluon


def net_initialize(net, model_ctx):
    net.collect_params().initialize(mx.init.Normal(sigma=.1), ctx=model_ctx)


def install_trainer(net):
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .01})
    return trainer


def epoch_loop(batch_loop):
    def decorader(
            net, begin_epoch, epoch_num,
            train_data,
            trainer, bp_loss_f,
            loss_function, smoothing_constant,
            model_ctx,
            inforer, timer,

    ):
        for epoch in range(begin_epoch, epoch_num):
            # initial
            moving_losses = {name: 0 for name in loss_function}
            inforer.batch_start(epoch)
            timer.start()
            batch_num, loss_values = batch_loop(
                net=net, epoch=epoch,
                train_data=train_data,
                trainer=trainer, bp_loss_f=bp_loss_f,
                loss_function=loss_function, moving_losses=moving_losses, smoothing_constant=smoothing_constant,
                model_ctx=model_ctx,
                batch_inforer=inforer,
            )
            inforer.batch_end(batch_num)


    return decorader


@epoch_loop
def batch_loop(fit_f):
    def decorader(
            net, epoch,
            train_data,
            trainer, bp_loss_f,
            loss_function, moving_losses, smoothing_constant,
            model_ctx,
            batch_inforer,
    ):
        # write loop body here
        for i, (data, label) in enumerate(train_data):
            fit_f(
                net=net, epoch=epoch, batch_i=i,
                data=data, label=label,
                trainer=trainer, bp_loss_f=bp_loss_f, loss_function=loss_function,
                moving_losses=moving_losses, smoothing_constant=smoothing_constant,
                model_ctx=model_ctx,
            )
            if i % 1 == 0:
                loss_values = [loss for loss in moving_losses.values()]
                batch_inforer.report(i, loss_value=loss_values)
        loss_values = {name: loss for name, loss in moving_losses.items()}.items()
        return loss_values, i
    return decorader


@batch_loop
def fit_f(net, epoch, batch_i,
          data, label,
          trainer, bp_loss_f, loss_function, moving_losses, smoothing_constant,
          model_ctx
          ):
    data = data.as_in_context(model_ctx)
    label = label.as_in_context(model_ctx)
    bp_loss = None
    with autograd.record():
        output = net(data)
        for name, func in loss_function.items():
            loss = func(output, label)
            if name in bp_loss_f:
                bp_loss = loss
            loss_value = nd.mean(loss).asscalar()
            moving_losses[name] = (loss_value if ((batch_i == 0) and (epoch == 0))
                                   else (1 - smoothing_constant) * moving_losses[
                name] + smoothing_constant * loss_value)

    assert bp_loss is not None
    bp_loss.backward()
    trainer.step(data.shape[0])

#######A#############
# candidates = int(input())
# votes = list(input().split())
#
# def run(candidates, votes):
#     limak_vote, others_votes = int(votes[0]), votes[1:]
#     others_votes = [int(others_vote) for others_vote in others_votes if int(others_vote) > limak_vote]
#     others_votes.sort(reverse=True)
#     bribes = 0
#     same = 0
#
#     def get_bribe(same, other_vote, vote):
#         if same == 0:
#             same = 1
#         bribe = (other_vote - vote + same) // (1 + same)
#         return bribe * same, vote + bribe * same
#
#     if len(others_votes) == 1:
#         print((others_votes[0] - limak_vote + 1) // 2)
#
#     for idx, vote in enumerate(others_votes):
#         if vote > limak_vote:
#             if idx + 1 < len(others_votes) and vote == others_votes[idx + 1]:
#                 same += 1
#             elif idx + 1 == len(others_votes):
#                 if vote == others_votes[idx - 1]:
#                     same += 1
#                 bribe, limak_vote = get_bribe(same, vote, limak_vote)
#                 bribes += bribe
#                 same = 0
#             else:
#                 bribe, limak_vote = get_bribe(same, vote, limak_vote)
#                 bribes += bribe
#                 same = 0
#         elif vote == limak_vote:
#             bribes += 1
#             return bribes
#         else:
#             return bribes
#
#     return bribes
# print(run(candidates, votes))


#####B#######
