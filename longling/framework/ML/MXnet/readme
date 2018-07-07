sudo apt-get install libatlas-base-dev

mxnet module fit arguments

for callback,
two kind:
    one is class, need __init__ method and __call__ method
    another is function, need return a functional object

# batch_end_callback(single argument: param)
## Speedometer

# epoch_end_callback(four arguments: iter_no, sym, arg, aux)
## do_checkpoint

# EvalMetric
## for F1 metric, exist serious problem



##################################################################
notice that embedding layer changes in training without grad block



###################################################################
lr_sch = mx.lr_scheduler.FactorScheduler(step=25000, factor=0.999)
    module.init_optimizer(
optimizer='rmsprop', optimizer_params={'learning_rate': 0.0005, 'lr_scheduler': lr_sch})