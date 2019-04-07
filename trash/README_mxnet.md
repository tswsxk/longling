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



### Notice
mxnet模块适合处理统一输入输出，采用已有评价体系的问题，尤其是分类问题
对于较为灵活的评价体系问题，较难处理，代表为TransE模型

#### Embedding
Embedding layer changes in training without grad block



### 其它
lr_sch = mx.lr_scheduler.FactorScheduler(step=25000, factor=0.999)
    module.init_optimizer(
optimizer='rmsprop', optimizer_params={'learning_rate': 0.0005, 'lr_scheduler': lr_sch})