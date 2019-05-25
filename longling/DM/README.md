# Feature Engineering

## feature-selector
https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650744875&idx=3&sn=e7de9ce4074077d4de1f40ab24ebe51f&chksm=871aec55b06d65430e9c54085e3933735a62465aabe5cc5227de11cde40ee831db9340858920&mpshare=1&scene=1&srcid=0707VGq05mvgqqfOLdX9E5ns&pass_ticket=k2xPEUKdPoigs428M2DOEBfREyoLVIxnT7agI9Jx33LvjVWxk38wrBeJqTWmC4TL#rd

https://github.com/WillKoehrsen/feature-selector/blob/master/feature_selector/feature_selector.py

## feature construct
https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650744044&idx=4&sn=098c8a6157cc814b4648449351e8e1cf&chksm=871ae092b06d6984f0f702c6d7857697f74717b1abd4e070859c3cf678c5e0996d75537f4e77&mpshare=1&scene=1&srcid=0621Gg6IwSWHzd1XdzlaTG5E&pass_ticket=k2xPEUKdPoigs428M2DOEBfREyoLVIxnT7agI9Jx33LvjVWxk38wrBeJqTWmC4TL#rd

https://docs.featuretools.com/

Paper: https://dai.lids.mit.edu/wp-content/uploads/2017/10/DSAA_DSM_2015.pdf

### Discover Feature Engineering, How to Engineer Features and How to Get Good at It
https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/

### The Protocol of Feature Engineering

https://www.zhihu.com/question/28641663

http://www.cnblogs.com/jasonfreak/p/5448462.html

### usful packages
https://zhuanlan.zhihu.com/p/42715527


### Steps of Data Mining
1. 数据采集
2. 数据预处理 DataPreProcess
    * 采样 DataSample
        * 针对大规模数据，抽取一小部分数据集用来跑测试
    * 数据分析
        * 特征分析 FeatureAnalysis
            * 分析特征所属类型
            * 数据类型包括
                * 定性特征
                    * 文本型，text，一般是长句子
                    * 类别型，可以是 text 或者 id
                * 定量特征：数值型，可以是 整型 或者 浮点型
                * 时间特征
    * 数据清洗 DataClean
        * 根据某些字段去掉某些数据
        * 修正某些数据错误
            * 多出的符号
    * 特征提取 FeatureExtract
        * 人工选取需要的特征 MankindFeature
        * 字段合并
7. 特征编码
    * 无量纲化
    * 定性特征：One-hot编码 OneHotEncoder
    * 定量特征
        * 二值化 Binarizer
        * 归一化 Normalizer
        * 区间缩放 MinMaxScaler
        * 标准化 StandardScaler
8. 特征预处理
    * 缺失值计算 Imputer
    * 数据变换 PolynomialFeatures FunctionTransformer
    * 噪声清洗 
9. 特征选择 VarianceThreshold SelectKBest SelectKBest+Chi2 SelectFromModel
10. 降维 PCA LDA
11. 训练
12. 预测
13. 评估