# coding: utf-8
# create by tongshiwei on 2017/10/5
from __future__ import print_function

import re
import sqlite3

from longling.base import *
from longling.lib.utilog import config_logging, LogLevel
from longling.lib.stream import rf_open, wf_open, wf_close

default_logger = config_logging(logger='dataBase', console_log_level=LogLevel.INFO, propagate=False)


class DataBase(object):
    def __init__(self, database_path, logger=default_logger):
        self.db = sqlite3.connect(database_path)
        self.cursor = self.db.cursor()
        self.logger = logger

    def set_logger_level(self, level):
        """设定数据库日志器的日志等级"""
        self.logger.setLevel(level)

    def get_table_info(self, table_name):
        """
        获取表信息

        Parameters
        ----------
        table_name: str

        Returns
        -------
        record_num: int
            表记录数
        create_statement: str
            创建语句
        """
        sql = "select * from sqlite_master where type='table' and name='%s'" % table_name
        self.cursor.execute(sql)
        info = self.fetchall()
        if not info:
            raise Exception("no such table")
        info = info[0]
        record_num, create_statement = info[3], info[4]
        return record_num, create_statement

    def get_column_name(self, table_name):
        """
        获取对应表的列名

        Parameters
        ----------
        table_name: str
            表名

        Returns
        -------
        column_names: list
            列名
        """
        _, create_statement = self.get_table_info(table_name)
        column_name = re.findall(r"\((.*)\)", create_statement)[0]
        return [name.split()[0] for name in column_name.split(',')]

    def execute(self, sql, args=''):
        try:
            if args:
                self.cursor.execute(sql, args)
            else:
                self.cursor.execute(sql)
        except Exception as e:
            self.logger.error("%s %s" % (sql, args))
            raise Exception(e)

    def fetchall(self, return_type=list):
        """
        取回所有数据

        Parameters
        ----------
        return_type: type(list) or type(dict)
            指定以何种形式返回数据

        Returns
        -------
        data: list or dict
            return_type指定格式的数据

        """
        if return_type == list:
            return self.cursor.fetchall()
        elif return_type == dict:
            keysName = self.fetch_field_name()
            return [dict(zip(keysName, values)) for values in self.cursor.fetchall()]

    def fetchmany(self, size=10000, return_type=list):
        """
        取回指定缓冲条目数的数据

        Parameters
        ----------
        size: int
            最大缓冲条目数
        return_type: type(list) or type(dict)
            指定以何种形式返回数据

        Returns
        -------
        data: list or dict
            return_type指定格式的数据，最多不超过size指定的数据条目

        """
        if return_type == list:
            return self.cursor.fetchmany(size)
        elif return_type == dict:
            keysName = self.fetch_field_name()
            return [dict(zip(keysName, values)) for values in self.cursor.fetchmany(size)]

    def fetch_field_name(self):
        """
        执行fetchall或fetchmany后可用，返回取出数据对应的列名

        Returns
        -------
        column_names: list
            fetchall或fetchmany操作取出的数据对应的列名
        """
        return [t[0] for t in self.cursor.description]

    def commit(self):
        """表修改操作后的许可操作"""
        self.db.commit()

    def close(self):
        self.cursor.close()
        self.db.close()

    def merge_on_sql(self, table1, table2, on_fields):
        if isinstance(on_fields, (list, tuple, set)):
            on_fields = set(on_fields)
        else:
            on_fields = {on_fields}

        def table_fields(table_name, fields_name):
            prefix = table_name + '.'
            return ",".join([prefix + name for name in fields_name])

        fields_name_1 = set(self.get_column_name(table1))
        fields_name_2 = set(self.get_column_name(table2))

        assert len(on_fields & fields_name_1) == len(on_fields)
        assert len(on_fields & fields_name_2) == len(on_fields)

        fields_name_2 -= on_fields

        table1_select = table_fields(table1, fields_name_1)
        table2_select = table_fields(table2, fields_name_2)

        on_sql = " and ".join(["%s.%s=%s.%s" % (table1, on_field, table2, on_field) for on_field in on_fields])

        sql = "select %s,%s from %s,%s where %s" % (table1_select, table2_select, table1, table2, on_sql)

        return sql

    def output_table_to_csv(self, tableName, csvPath, batchSize=10000):
        """
        将数据库内的表导出为csv格式

        Parameters
        ----------
        tableName: str
            要导出的的表的表名
        csvPath: str
            导出以逗号分隔的文件路径
        batchSize: int
            缓冲区条目数
        """

        sql = "select * from %s" % tableName
        self.execute(sql)
        fieldName = self.fetch_field_name()
        self.db.text_factory = str
        self.logger.info("output_table_to_csv: %s-%s" % (tableName, fieldName))
        wf = wf_open(csvPath)
        print(",".join(fieldName), file=wf)
        cnt = 0
        datas = self.fetchmany(batchSize)

        while datas:
            print("\n".join([",".join([d.decode('utf-8') for d in data]) for data in datas]), file=wf)
            cnt += len(datas)
            self.logger.debug("output_table_to_csv: %s-%s" % (tableName, cnt))
            datas = self.fetchmany()
        wf_close(wf)
        self.logger.info("output_table_to_csv: output %s to %s-%s" % (tableName, csvPath, cnt))

    def import_csv_to_databse(self, csvPath, tableName, batchSize=50000):
        """
        向数据库中导入表（原格式csv, 逗号分隔），如有列名重复，取第一次出现的

        Parameters
        ----------
        csvPath: str
            逗号分隔的含列名数据文件路径
        tableName: str
            要导出的的表的表名
        batchSize: int
            缓冲区条目数

        Returns
        -------
        cnt: int
            导入数据条目数
        """
        try:
            self.logger.info("import_csv_to_databse: clean exsiting table %s" % tableName)
            sql = "drop table %s" % tableName
            self.execute(sql)
            self.commit()
            self.logger.debug("import_csv_to_databse: drop table %s" % tableName)
        except:
            pass

        with rf_open(csvPath) as f:
            keys = f.readline().strip().split(",")
            df = DupColumnFilter(keys)
            keys = df.get_keys()
            keys_desc = ",".join([key + ' text' for key in keys])
            sql = "create table %s (%s)" % (tableName, keys_desc)
            self.execute(sql)
            self.commit()

            placeholder = '(' + ",".join(['?'] * len(keys)) + ')'
            sql = "insert into %s values %s" % (tableName, placeholder)
            self.logger.info("import_csv_to_databse: start import csv-%s to table-%s" % (csvPath, tableName))
            buff = []
            cnt = 0
            for line in f:
                line = unistr(line)
                line = line.strip()
                if line:
                    line = line.split(",")
                    line = df.filter_values(line)
                    if len(line) != len(keys):
                        self.logger.warning("Incorrect number of bindings supplied: %s" % line)
                        self.logger.warning(
                            "Incorrect number of bindings supplied. "
                            "The current statement uses %s, but there are %s supplied"
                            % (len(keys), len(line)))
                        if len(line) > len(keys):
                            self.logger.warning("%s values in line will be abandoned" % (len(line) - len(keys)))
                            line = line[:len(keys)]
                        else:
                            self.logger.warning("only %s values are supplied, this record will be abandoned" % len(line))
                            continue
                    buff.append(line)
                    cnt += 1
                    if cnt % batchSize == 0:
                        self.db.executemany(sql, buff)
                        self.commit()
                        self.logger.debug("import_csv_to_databse: %s-%s" % (tableName, cnt))
                        buff = []
            if buff:
                self.db.executemany(sql, buff)
                self.commit()
                self.logger.debug("%s-%s" % (tableName, cnt))
            self.logger.info("import_csv_to_databse: import %s to %s-%s" % (csvPath, tableName, cnt))
            return cnt


class DupColumnFilter(object):
    def __init__(self, keys):
        self.dupColumnIndex = set()
        self.keys = []
        column_names_set = set()
        for i, key in enumerate(keys):
            if key not in column_names_set:
                column_names_set.add(key)
                self.keys.append(key)
            else:
                self.dupColumnIndex.add(i)

    def filter_values(self, values):
        if not self.dupColumnIndex:
            return values
        res = []
        for i, value in enumerate(values):
            if i not in self.dupColumnIndex:
                res.append(value)
        return res

    def get_keys(self):
        return self.keys
