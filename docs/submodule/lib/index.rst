lib: General Library
====================


Quick Glance
-------------
For io:

.. autosummary::
   longling.lib.stream.to_io
   longling.lib.stream.as_io
   longling.lib.stream.as_out_io
   longling.lib.loading.loading

迭代器

.. autosummary::
   longling.lib.iterator.AsyncLoopIter
   longling.lib.iterator.CacheAsyncLoopIter
   longling.lib.iterator.iterwrap

日志

.. autosummary::
   longling.lib.utilog.config_logging

For path

.. autosummary::
   longling.lib.path.path_append
   longling.lib.path.abs_current_dir
   longling.lib.path.file_exist


语法糖

.. autosummary::
   longling.lib.candylib.as_list


计时与进度

.. autosummary::
   longling.lib.clock.print_time
   longling.lib.clock.Clock
   longling.lib.stream.flush_print


并发

.. autosummary::
   longling.lib.concurrency.concurrent_pool


测试

.. autosummary::
   longling.lib.testing.simulate_stdin

结构体
.. autosummary::
   longling.lib.structure.AttrDict
   longling.lib.structure.nested_update
   longling.lib.structure.SortedList

正则
.. autosummary::
   longling.lib.structure.variable_replace
   longling.lib.structure.default_variable_replace


candylib
-------------
.. automodule:: longling.lib.candylib
   :members:
   :imported-members:

clock
-------------
.. automodule:: longling.lib.clock
   :members:
   :imported-members:

concurrency
-----------
.. automodule:: longling.lib.concurrency
   :members:
   :imported-members:


formatter
-----------
.. automodule:: longling.lib.formatter
   :members:
   :imported-members:

iterator
--------
.. automodule:: longling.lib.iterator
   :members:
   :imported-members:

loading
-------------
.. automodule:: longling.lib.loading
   :members:
   :imported-members:

parser
-------------
自定义的配置文件及对应的解析工具包。目的是为了更方便、快速地进行文件参数配置与解析。

.. automodule:: longling.lib.parser
   :members:
   :imported-members:

path
-------------
.. automodule:: longling.lib.path
   :members:
   :imported-members:

progress
-------------
进度监视器，帮助用户知晓当前运行进度，主要适配于机器学习中分 epoch，batch 的情况。

和 tqdm 针对单个迭代对象进行快速适配不同，
progress的目标是能将监视器不同功能部件模块化后再行组装，可以实现description的动态化，
给用户提供更大的便利性。

* MonitorPlayer 定义了如何显示进度和其它过程参数(better than tqdm, where only n is changed and description is fixed)
    * 在 __call__ 方法中定义如何显示
* 继承ProgressMonitor并传入必要参数进行实例化
    * 继承重写ProgressMonitor的__call__函数，用 IterableMIcing 包裹迭代器，这一步可以灵活定义迭代前后的操作
    * 需要在__init__的时候传入一个MonitorPlayer实例
* IterableMIcing 用来组装迭代器、监控器

一个简单的示例如下

.. code-block:: python

    class DemoMonitor(ProgressMonitor):
        def __call__(self, iterator):
            return IterableMIcing(
                iterator,
                self.player, self.player.set_length
            )

    progress_monitor = DemoMonitor(MonitorPlayer())

    for _ in range(5):
        for _ in progress_monitor(range(10000)):
            pass
        print()

cooperate with tqdm

.. code-block:: python

    from tqdm import tqdm

    class DemoTqdmMonitor(ProgressMonitor):
        def __call__(self, iterator, **kwargs):
            return tqdm(iterator, **kwargs)

.. automodule:: longling.lib.progress
   :members:
   :imported-members:

regex
---------------
.. automodule:: longling.lib.regex
   :members:
   :imported-members:

stream
-------------
.. automodule:: longling.lib.stream
   :members:
   :imported-members:

structure
-------------
.. automodule:: longling.lib.structure
   :members:
   :imported-members:

testing
-------
.. automodule:: longling.lib.testing
   :members:
   :imported-members:

time
-------------
.. automodule:: longling.lib.time
   :members:
   :imported-members:

utilog
-------------
.. automodule:: longling.lib.utilog
   :members:
   :imported-members:

yaml
-----
.. automodule:: longling.lib.yaml_helper
   :members:
   :imported-members:
