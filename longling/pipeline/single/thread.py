# coding: utf-8
# create by tongshiwei on 2019/5/26

import os
import threading


class ThreadGroup(list):
    def add(self, thread):
        self.append(thread)

    def join(self):
        for thread in self:
            thread.join()

    def __enter__(self, *args, **kwargs):
        return ThreadGroup(*args, **kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.join()

def async_remove(filename, thread_group):
    thread = threading.Thread(
        target=os.remove,
        args=[filename, ]
    )
    thread_group.add(thread)
