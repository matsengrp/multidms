=========
Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com>`_.

1.1.0
-----
* No longer calling transform() on parameters for single condition fits. See `#160 <https://github.com/matsengrp/multidms/issues/160>`_.
* Added `init_beta_variance` parameter to the `Model` instantiation to allow the user to initialize beta parameters by sampling a normal distribution. See `#161 <https://github.com/matsengrp/multidms/issues/161>`_.


1.0.0
-----
- This release re-implements the joint model as a using a generalized lasso, and bit-flipping, as described in `#156 <https://github.com/matsengrp/multidms/issues/156>`_. Please see the issue for more detailed description about how, and why these changes were made. Note that this changes the parameters that one may get from the model including a set of beta's for each experimental condition.
- It also cleans up various TODO's in the code as checked-off in `#153 <https://github.com/matsengrp/multidms/issues/153>`.
- Fixes a bug, where the phenotype predictions for single mutants did not correctly include the bundle effects.
- Fixes and cleans various plotting bugs. 

0.4.0
-----
- new simulation validation analysis and plotting functions (at the time of re-submission)
- fixes bug described in `#130 https://github.com/matsengrp/multidms/issues/130`_, having to do with pandas groupby.apply 2.2.0 behavior change.
- updates python version requirements to 3.9 or newer, as 3.8 did not work with the new pandas version, 2.2.0 bug patch described above.
- supresses the cpu warning from jax.
- adds `ModelCollection.add_validation_loss <https://github.com/matsengrp/multidms/blob/b0e7cbe96216e1307d070adc531fe51a960ec32a/multidms/model_collection.py#L569>`_, `ModelCollection.get_conditional_loss_df <https://github.com/matsengrp/multidms/blob/b0e7cbe96216e1307d070adc531fe51a960ec32a/multidms/model_collection.py#L627>`_, `Model.conditional_loss <https://github.com/matsengrp/multidms/blob/b0e7cbe96216e1307d070adc531fe51a960ec32a/multidms/model.py#L379>`_, and `Model.get_df_loss <https://github.com/matsengrp/multidms/blob/b0e7cbe96216e1307d070adc531fe51a960ec32a/multidms/model.py#L568>`_ methods, which can all be used quite easily to perform cross validation analysis.

0.3.3
-----
- simply updates the ruff linting to version `0.0.289`

0.3.2
-----
- fixes bug `#128 <https://github.com/matsengrp/multidms/issues/128>`_

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
