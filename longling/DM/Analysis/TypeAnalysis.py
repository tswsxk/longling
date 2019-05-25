# coding: utf-8
# create by tongshiwei on 2019/5/25

from longling.DM.structure.Encoder import OrdinalEncoder


class StaticsOrdinalEncoder(OrdinalEncoder):
    """
    将特征转换为定性特征值，每一种特征都当成一类。在fit过程中统计类频率。
    """

    def __init__(self, mapper_meta=None):
        super(StaticsOrdinalEncoder, self).__init__(mapper_meta)
        self.counter = {key: 0 for key in self.mapper}

    def add(self, values):
        super(StaticsOrdinalEncoder, self).add(values)
        for value in self.as_list(values):
            if values not in self.counter:
                self.counter[value] = 0
            self.counter[value] += 1

