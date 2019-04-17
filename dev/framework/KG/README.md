# 关系机器学习
## 基于翻译的embedding模型
### 需要考虑的几个问题
1. 实体和关系是否共享embedding
    - subject 和 object 共享 实体embedding，关系单独使用 关系embedding
    - subject, relation, object 各自使用自己的embedding
    - subject, relation, object 共享使用统一的embedding

2. 数据集的表示形式
    生肉数据集一般是字符串，直接构建训练、测试集会使得存储空间极大，所以需要先转成数字来存储
    所以需要以下几个文件：
    - 映射文件（可能有多个），记录了字符串到数字的映射关系
    - 映射信息文件（可选），记录了每种映射的基本信息，例如总长度等
    
    处理的时候一般先生成映射，并讲映射关系存储起来，再对每个文件进行映射转换，转换后再进行相应操作

### 评价指标
- Mean-Rank: 正样本在所有样本（多个负样本+这个正样本）按分数排序中的平均排位
- hits@n (n=10): 正样本在所有样本（多个负样本+这个正样本）按分数排序中排在前n位的比例
#### 数据划分
##### 根据关系类型划分 （hits@n）
- 1 to 1: 在给定关系下，一个 subject 只对应一个 object
- 1 to n: 在给定关系下，一个 subject 对应多个 object
- n to 1: 在给定关系下，一个 object 对应多个 subject
- n to n: 在给定关系下，多个 subject 对应多个 object

分别统计对应关系，然后计算在给定关系下
- sub_objn: 单个 subject 对应的 object 数
- obj_subn: 单个 object 对应的 subject 数
并计算 avg_sub_objn 和 avg_obj_subn
- 1 to 1: avg_sub_objn \<= 1.5 and avg_obj_subn \<= 1.5
- 1 to n: avg_sub_objn \> 1.5 and avg_obj_subn \<= 1.5
- n to 1: avg_sub_objn \<= 1.5 and avg_obj_subn \> 1.5
- n to n: avg_sub_objn \> 1.5 and avg_obj_subn \> 1.5

注意:
1. 只需要对test进行重新划分
2. 关系类别分类的时候需要将三个集合整合在一起
3. predicating tail 指的是给定subject, relation, 从 objects 中找出评分最高的(sub, rel, obj) 中的object
##### New relationships prediction (Mean-Rank & hits@n)
1. 将数据集划分为两类
    - 待预测的 n 种关系(记为 n-rel), 每种关系包含 m 个三元组样本，并进一步分为 n-rel-train 和 n-rel-valid, 
    两者互为补集, n-rel-train 的每种关系包含 k 个三元组样本(即 n-rel-valid 的每种关系只包含 m - k 个三元组样本)
    - 除上述 n 种关系外的其它关系(记为n-rest), 每种关系包含所有含该关系的三元组样本，
    并进一步分为 n-rest-train, n-rest-valid
2. 模型训练与评价
    - 在 n-rest-train 上进行训练，在 n-rest-valid 进行评估选取
    - 设定 k 值, 在 n-rel-train 上继续训练
    - 在 n-rel-valid 得到评估结果
3. 注意：这个划分需要重新将train-test-valid组合起来重新做