# coding: utf-8
# create by tongshiwei on 2017/10/9

'''
此模块用来把数据从数据源转换成一个迭代器格式，方便后续的数据处理统一化
'''

from __future__ import absolute_import
from longling.framework.ML.universe.dataIterator import OriginBatchIterator


class DBBatchIterator(OriginBatchIterator):
    def __init__(self, db, sql, batch=100000):
        self.sql = sql
        self.db = db
        self.db.execute(self.sql)
        self.batch = batch

    def get_field_name(self):
        return self.db.fetch_field_name()

    def next_batch(self):
        try:
            datas = self.db.fetchmany(self.batch)
            if datas:
                return datas
            else:
                raise StopIteration
        except StopIteration:
            raise StopIteration

    def next(self):
        return self.next_batch()


class DBIterator(DBBatchIterator):
    def __init__(self, db, sql):
        super(DBIterator).__init__(db, sql)
        self.index = 0
        self.batch_buff = []

    def next(self):
        if self.index == len(self.batch_buff):
            self.batch_buff = self.next_batch()
            self.index = 0
        else:
            data = self.batch_buff[self.index]
            self.index += 1
            return data

