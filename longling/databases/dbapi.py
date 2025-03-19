# coding: utf-8
# tongshiwei@2025/1/28

from datetime import datetime
from abc import ABC, abstractmethod


class DBAPI(ABC):
    @abstractmethod
    def execute(self, sql, *args, **kwargs):
        raise NotImplementedError

    def execute_many(self, sqls, *args, **kwargs):
        raise NotImplementedError

    def get_schema(self, schema) -> dict:
        raise NotImplementedError

    def insert(self, *args, **kwargs):
        raise NotImplementedError

    def insert_dataframe(self, *args, **kwargs):
        raise NotImplementedError

    def delete(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def select(self, *args, **kwargs):
        raise NotImplementedError

    def connect(self, *args, **kwargs):
        raise  NotImplementedError


class SQLMixIn(object):
    @classmethod
    def format_where_clause(cls, values: dict):
        """

        Parameters
        ------
        values:

        Returns
        -------

        Examples
        ---------
        >>> SQLMixIn.format_where_clause({"key1": ["a", "b", "c"], "key2": "d"})
        "WHERE key1 in ('a', 'b', 'c') AND key2 = 'd'"
        """
        return f"WHERE {cls.format_sql_value(values, concatenate_char=' AND ')}"

    @classmethod
    def format_sql_value(cls, value, add_quotation=True, concatenate_char=", ") -> str:
        """
        Format and generate SQL query values.

        Parameters
        ----------
        value : any
            The value to format, which can be a dictionary, list, tuple, datetime, or other types.
        add_quotation : bool, optional
            Whether to add quotes around string values (default is True).
        concatenate_char : str, optional
            The character to use for concatenating key-value pairs when the value is a dictionary (default is ", ").

        Returns
        -------
        str
            The formatted SQL query value.

        Examples
        --------
        >>> SQLMixIn.format_sql_value("test")
        "'test'"
        >>> SQLMixIn.format_sql_value(123)
        "123"
        >>> SQLMixIn.format_sql_value(123.45)
        "123.45"
        >>> SQLMixIn.format_sql_value(None)
        "None"
        >>> SQLMixIn.format_sql_value(["a", "b", "c"])
        "('a', 'b', 'c')"
        >>> SQLMixIn.format_sql_value((1, 2, 3))
        "(1, 2, 3)"
        >>> from datetime import datetime
        >>> dt = datetime(2025, 1, 29, 12, 30, 45, 123456)
        >>> SQLMixIn.format_sql_value(dt)
        "'2025-01-29 12:30:45.123456'"
        """
        if isinstance(value, dict):
            return concatenate_char.join(
                f"{k} {'in' if isinstance(v, list) else '='} "
                f"{cls.format_sql_value(v)}"
                for k, v in value.items())
        elif isinstance(value, (list, tuple)):
            return f'({", ".join(map(cls.format_sql_value, value))})'
        elif isinstance(value, datetime):
            value = value.strftime("%Y-%m-%d %H:%M:%S.%f")
        elif value is None:
            value = str(value)
        value = value.replace("'", "''")
        if add_quotation:
            return f"""'{value}'"""
        else:
            return value
