# coding: utf-8
# tongshiwei@2025/1/28

import inspect

from loguru import logger as logging
from sqlalchemy import create_engine, inspect as sql_inspect, text

from .dbapi import DBAPI


class SqlAlchemyAPI(DBAPI):
    def __init__(self, uri, **kwargs):
        self.uri = uri
        self.engine = create_engine(self.uri, **kwargs)
        self.inspector = sql_inspect(self.engine)

    def execute(self, sql, commit=False,  normalize_sql=True, *args, **kwargs):
        if normalize_sql:
            sql = inspect.cleandoc(sql)
            # since not using connection.execute(sql, kwargs), escape colons
            sql = sql.replace("\\", "\\\\").replace(":", "\\:")

        with self.engine.connect() as connection:
            logging.debug(f"execute:\n{sql}")
            cursor = connection.execute(text(sql))

            if commit:
                connection.commit()
                return cursor
            else:
                return cursor.fetchall()
