# coding: utf-8

from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from tqdm import tqdm

from longling.framework.db.sqlite import DataBase
from longling.framework.ML.universe.dataIterator import DBIterator
from longling.lib.stream import wf_open, wf_close


def transform_db2csv(database_path, sql, target):
    db = DataBase(database_path)
    datas = DBIterator(db, sql)
    fields_name = datas.get_field_name()

    wf = wf_open(target)
    print(",".join(fields_name), file=wf)
    for data in tqdm(datas):
        print(",".join(data), file=wf)
    wf_close(wf)
