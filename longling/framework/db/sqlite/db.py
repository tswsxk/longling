# coding: utf-8
# create by tongshiwei on 2017/10/5
from __future__ import print_function

import logging
import re
import sqlite3

from longling.base import *
from longling.lib.utilog import config_logging
from longling.lib.stream import *

logger = logging.getLogger('dataBase')
config_logging(logger=logger, level=logging.DEBUG, console_log_level=logging.DEBUG, propagate=False)


class dataBase():
    def __init__(self, dataBasePath, logLevel=logging.DEBUG):
        self.db = sqlite3.connect(dataBasePath)
        self.cursor = self.db.cursor()
        logger.setLevel(logLevel)

    def set_logger_level(self, level):
        logger.setLevel(level)

    def get_table_info(self, table_name):
        sql = "select * from sqlite_master where type='table' and name='%s'" % table_name
        self.cursor.execute(sql)
        info = self.fetchall()
        if not info:
            raise Exception("no such table")
        info = info[0]
        record_num, create_statement = info[3], info[4]
        return record_num, create_statement

    def get_column_name(self, table_name):
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
            logger.info("%s %s" % (sql, args))
            raise Exception(e)

    def fetchall(self, return_type=list):
        if return_type == list:
            return self.cursor.fetchall()
        elif return_type == dict:
            keysName = self.fetch_fieldName()
            return [dict(zip(keysName, values)) for values in self.cursor.fetchall()]

    def fetchmany(self, size=10000, return_type=list):
        if return_type == list:
            return self.cursor.fetchmany(size)
        elif return_type == dict:
            keysName = self.fetch_fieldName()
            return [dict(zip(keysName, values)) for values in self.cursor.fetchmany(size)]

    def fetch_fieldName(self):
        return [t[0] for t in self.cursor.description]

    def commit(self):
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
        '''
        将数据库内的表导出为csv格式
        :param tableName:
        :param csvPath:
        :param batchSize:
        :return:
        '''
        sql = "select * from %s" % tableName
        self.execute(sql)
        fieldName = self.fetch_fieldName()
        self.db.text_factory = str
        logger.info("output_table_to_csv: %s-%s" % (tableName, fieldName))
        wf = wf_open(csvPath)
        print(",".join(fieldName), file=wf)
        cnt = 0
        datas = self.fetchmany(batchSize)

        while datas:
            print("\n".join([",".join([d.decode('utf-8') for d in data]) for data in datas]), file=wf)
            cnt += len(datas)
            logger.debug("output_table_to_csv: %s-%s" % (tableName, cnt))
            datas = self.fetchmany()
        wf_close(wf)
        logger.info("output_table_to_csv: output %s to %s-%s" % (tableName, csvPath, cnt))

    def import_csv_to_databse(self, csvPath, tableName, batchSize=50000):
        '''
        向数据库中导入表（原格式csv），如有列名重复，取第一次出现的
        :param csvPath:
        :param tableName:
        :param batchSize:
        :return:
        '''
        try:
            logger.info("import_csv_to_databse: clean exsiting table %s" % tableName)
            sql = "drop table %s" % tableName
            self.execute(sql)
            self.commit()
            logger.debug("import_csv_to_databse: drop table %s" % tableName)
        except:
            pass

        with rf_open(csvPath) as f:
            keys = f.readline().strip().split(",")
            df = dupColumnFilter(keys)
            keys = df.get_keys()
            keys_desc = ",".join([key + ' text' for key in keys])
            sql = "create table %s (%s)" % (tableName, keys_desc)
            self.execute(sql)
            self.commit()

            placeholder = '(' + ",".join(['?'] * len(keys)) + ')'
            sql = "insert into %s values %s" % (tableName, placeholder)
            logger.info("import_csv_to_databse: start import csv-%s to table-%s" % (csvPath, tableName))
            buff = []
            cnt = 0
            for line in f:
                line = unistr(line)
                line = line.strip()
                if line:
                    line = line.split(",")
                    line = df.filtValues(line)
                    if len(line) != len(keys):
                        logger.warning("Incorrect number of bindings supplied: %s" % line)
                        logger.warning(
                            "Incorrect number of bindings supplied. "
                            "The current statement uses %s, but there are %s supplied"
                            % (len(keys), len(line)))
                        if len(line) > len(keys):
                            logger.warning("%s values in line will be abandoned" % (len(line) - len(keys)))
                            line = line[:len(keys)]
                        else:
                            logger.warning("only %s values are supplied, this record will be abandoned" % len(line))
                            continue
                    buff.append(line)
                    cnt += 1
                    if cnt % batchSize == 0:
                        self.db.executemany(sql, buff)
                        self.commit()
                        logger.debug("import_csv_to_databse: %s-%s" % (tableName, cnt))
                        buff = []
            if buff:
                self.db.executemany(sql, buff)
                self.commit()
                logger.debug("%s-%s" % (tableName, cnt))
            logger.info("import_csv_to_databse: import %s to %s-%s" % (csvPath, tableName, cnt))
            return cnt


class dupColumnFilter():
    def __init__(self, keys):
        self.dupColumnIndex = set()
        self.keys = []
        columnNamesSet = set()
        for i, key in enumerate(keys):
            if key not in columnNamesSet:
                columnNamesSet.add(key)
                self.keys.append(key)
            else:
                self.dupColumnIndex.add(i)

    def filtValues(self, values):
        if not self.dupColumnIndex:
            return values
        res = []
        for i, value in enumerate(values):
            if i not in self.dupColumnIndex:
                res.append(value)
        return res

    def get_keys(self):
        return self.keys

