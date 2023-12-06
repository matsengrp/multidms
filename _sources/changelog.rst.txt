=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com>`_.

0.3.1
-----
- fixes bug `#126 <https://github.com/matsengrp/multidms/issues/126>`_.
- Adds the initial working simulation notebook.


0.3.0
-----
- Adds initial `multidms.model_collection <https://github.com/matsengrp/multidms/blob/main/multidms/model_collection.py>`_ module with ``multidms.fit_models`` for the ability to fit multiple models across a range of parameter spaces in parallel using `multiprocessing`. This is inspired by the `polyclonal.fit_models` function. 
- Adds the ``ModelCollection`` class for split-apply-combine interface to the mutational dataframes for a collection of models
- Adds four altair plotting methods to ``ModelCollection``: ``mut_param_heatmap``, ``shift_sparsity``, ``mut_param_dataset_correlation``, and ``mut_param_traceplot``.
- removes ``utils`` module.
- Cleans up #114 
- optionally removes "wts", "sites", "muts" from the mutations dataframe returned by ``Model.get_mutations_df``. 
- Changes the naming of columns produced by ``Model.get_mutations_df()``, in particular, it moves the condition name for predicted func score to be a suffix (as with shift, and time_seen) rather than a prefix. e.g. "delta_predicted_func_score" -> "predicted_func_score_delta".


0.2.2
-----
- Fixed a `bug <https://github.com/matsengrp/multidms/issues/116>`_ 
    caused by non-unique indices in input variant functional score dataframes.


0.2.1
-----
- Made lineplot_and_heatmap() more private to remove from docs.
- Fixed bug pointed out by @jbloom #110
- ``Model.get_mutations_df()`` now sets the mutation as the index
- added some testing utils

0.2.0
-----
- Closed a `docs testing issue <https://github.com/matsengrp/multidms/issues/104>`_, thanks, @WSDeWitt !
- Cleaned Actions, again thanks to @WSDeWitt
- Fixed a `bug in wildtype predictions <https://github.com/matsengrp/multidms/issues/106>`_
- Implemented `QC on invalid bundle muts <https://github.com/matsengrp/multidms/issues/84>`_ as pointed out by @Haddox.
- a few other minor cleanup tasks.


0.1.9
-----
- First Release on PYPI 