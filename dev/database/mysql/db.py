# coding: utf-8
import os, threading, re
import MySQLdb

'''
brew install mysql
pip install MySQL-python
'''

class DAL(threading.local):
    dry_run = False

    def __init__(self, host, name, user, passwd, port=3306, connect_timeout=1, read_timeout=0, charset='utf8'):
        # 对于list类型host，拼成字符串
        host = ','.join(host) if type(host) is list else host
        self.host, self.port, self.name, self.user, self.passwd = host, int(port), name, user, passwd
        self.conn_key = '%s|%s:%s@%s:%s/%s' % (os.getpid(), user, passwd, host, port, name)
        self.cursor = None
        self.conn = None
        self.charset = charset
        # unit: second
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout

    def open(self):
        global local_conn
        if not local_conn:
            local_conn = threading.local()
        if not hasattr(local_conn, 'connections'):
            local_conn.connections = {}
        conn = local_conn.connections.get(self.conn_key)
        if not conn:
            conn = MySQLdb.connect(host=self.host, port=self.port, user=self.user, passwd=self.passwd,
                                   db=self.name, charset=self.charset, autocommit=True,
                                   connect_timeout=self.connect_timeout,
                                   read_timeout=self.read_timeout,
                                  )
            local_conn.connections[self.conn_key] = conn
        self.conn = conn
        self.cursor = conn.cursor(cursorclass=MySQLdb.cursors.DictCursor)

    def get_cursor(self):
        if not self.cursor:
            self.open()
        return self.cursor

    def execute(self, sql_fmt, *params, **kwargs):
        retries = kwargs.get('retries', 1)
        while True:
            try:
                cursor = self.get_cursor()
                if self.dry_run:
                    m = re.search(r'^\s*(update|insert|replace|delete)', sql_fmt, re.I)
                    if m:
                        return cursor
                cursor.execute(sql_fmt, params or None)
                return cursor
            except:
                self.close()
                if retries <= 0:
                    raise
                retries -= 1

    def close(self):
        global local_conn
        if self.cursor:
            self.cursor.close()
            self.cursor = None

        if self.conn:
            conn = local_conn.connections.pop(self.conn_key, None)
            if conn:
                conn.close()
            self.conn = None

if __name__ == '__main__':
    print "x"
