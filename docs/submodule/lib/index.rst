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
进度监视器，帮助用户知晓当前运行进度

一个简单的示例如下

.. code-block:: python

    class DemoMonitor(ProgressMonitor):
        def __call__(self, iterator):
            return IterableMonitor(
                iterator,
                self.player, self.player.set_length
            )

    progress_monitor = DemoMonitor(MonitorPlayer())

    for _ in range(5):
        for _ in progress_monitor(range(10000)):
            pass
        print()

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
.. automodule:: longling.lib.yaml
   :members:
   :imported-members:
