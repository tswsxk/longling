Command Line Interfaces
=======================
use ``longling -- --help`` to see all available cli,
and use ``longling $subcommand -- --help`` to see concrete help information for a certain cli,
e.g. ``longling encode -- --help``

Format and Encoding
-------------------
.. autosummary::
   longling.lib.stream.encode
   longling.lib.loading.csv2json
   longling.lib.loading.json2csv


Download Data
-------------
.. autosummary::
   longling.spider.download_data.download_file

Architecture
------------
.. autosummary::
   longling.toolbox.toc.toc
   longling.Architecture.cli.cli
   longling.Architecture.install_file.nni
   longling.Architecture.install_file.gitignore
   longling.Architecture.install_file.pytest
   longling.Architecture.install_file.coverage
   longling.Architecture.install_file.pysetup
   longling.Architecture.install_file.sphinx_conf
   longling.Architecture.install_file.makefile
   longling.Architecture.install_file.readthedocs
   longling.Architecture.install_file.travis
   longling.Architecture.install_file.dockerfile
   longling.Architecture.install_file.gitlab_ci
   longling.Architecture.install_file.chart

Specials
^^^^^^^^


Model Selection
---------------
Validation on Datasets
^^^^^^^^^^^^^^^^^^^^^^^
Split dataset to train, valid and test or apply kfold.

.. autosummary::
   longling.ML.toolkit.dataset.train_valid_test
   longling.ML.toolkit.dataset.train_valid
   longling.ML.toolkit.dataset.train_test
   longling.ML.toolkit.dataset.kfold

Select Best Model
^^^^^^^^^^^^^^^^^
Select best models on specified conditions

.. autosummary::
   longling.ML.toolkit.analyser.cli.select_max
   longling.ML.toolkit.analyser.cli.arg_select_max
   longling.ML.toolkit.hyper_search.nni.show_top_k
   longling.ML.toolkit.hyper_search.nni.show
