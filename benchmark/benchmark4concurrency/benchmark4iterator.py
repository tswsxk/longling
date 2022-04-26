# coding: utf-8
# 2022/4/26 @ tongshiwei

from tqdm import tqdm
from longling import iterwrap, print_time

SAMPLE_NUM = int(5e3)
PROCESS_LOAD = 500  # When Process_LOAD is small, thread will be more appropriate than process


def cpu_bound_func(a):
    for _ in range(PROCESS_LOAD):
        a += 1
    return a


@iterwrap()
def etl_thread():
    for _ in range(SAMPLE_NUM):
        ret = []
        for _ in range(16):
            ret.append(cpu_bound_func(0))
        yield ret


@iterwrap(level="p")
def etl_process():
    for _ in range(SAMPLE_NUM):
        ret = []
        for _ in range(16):
            ret.append(cpu_bound_func(0))
        yield ret


if __name__ == '__main__':
    data = etl_thread()
    with print_time("t"):
        for e in range(2):
            for i in tqdm(data, "%s" % e):
                x = [cpu_bound_func(j) for j in i]

    data = etl_process()
    with print_time("p"):
        for e in range(2):
            for i in tqdm(data, "%s" % e):
                x = [cpu_bound_func(j) for j in i]
