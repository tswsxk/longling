#
## 两种构建方式的比较

### gluon.nn.HybridSequential()
只能序列化，不能自定义结构之间是如何连接的

### 类继承 gluon.HybridBlock
另一种则可以

## gluon.HybridBlock

### hybrid_forward
#### 第一个参数 F
等价于 mxnet.ndarray

