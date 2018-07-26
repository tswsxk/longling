# batchify 的原则
1. 数据量不大，可以全部转载到内存中，统一使用 gluon.data.DataLoader 来装载
    * 对长度一致数据，可以用 gluon.data.ArrayDataset 来直接装载
    * 对长度不一致数据，可以用Pad方法先进行补齐, 或者用Clip方法截断
    * 或者使用 Bucket 的方式来进行分桶（bucketing）
    * trick 可以用分离出数据index的方式来进行sample和shuffle及bucket操作
2. 数据量很大的情况下，可能不能一次性全部装载到内存里
    * 这些操作需要在文件级上进行，生成中间文件结果：排序（Sort）
    * 这些操作在内存（装载-提供数据）时进行：批量化(batchify)
    * 这些操作可以选择在文件或内存（装载-提供数据）时进行：补齐（Pad）
    * 这些操作尽量后延到最后提供数据时（内存）进行，以压缩文件存储空间：Embedding
    按顺序看：
    1. 排序
    2. 分批 & 补齐
    3. Embedding
    对不同操作而言
    * 分桶：排序 -> 分批 -> 补齐