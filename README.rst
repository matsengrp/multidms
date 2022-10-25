===============================
multidms
===============================

.. todo:: code style badge
.. todo:: docker build badge
.. todo:: gh actions build and test
.. todo:: documentation

Model one or more dms experiments
and identify variant predicted fitness, and 
individual mutation effects and shifts.

``multidms`` is a Python package written by `the Bloom lab <https://research.fhcrc.org/bloom/en.html>`_.

See `Yu et al (2022) <https://www.biorxiv.org/content/10.1101/2022.09.17.508366v1>`_ for an explanation of the approach implemented in ``multidms``.

The source code is `on GitHub <https://github.com/matsengrp/multidms>`_.

See the `multidms documentation <https://matsengrp.github.io/multidms>`_ for details on how to install and use ``multidms``.

To contribute to this package, read the instructions in `CONTRIBUTING.rst <CONTRIBUTING.rst>`_.

>>> import pandas as pd
>>> import multidms
>>> func_score_data = {
...     'condition' : ["1","1","1","1", "2","2","2","2","2","2"],
...     'aa_substitutions' : ['M1E', 'G3R', 'G3P', 'M1W', 'M1E', 'P3R', 'P3G', 'M1E P3G', 'M1E P3R', 'P2T'],
...     'func_score' : [2, -7, -0.5, 2.3, 1, -5, 0.4, 2.7, -2.7, 0.3],
... }
>>> func_score_df = pd.DataFrame(func_score_data)
>>> func_score_df
  condition aa_substitutions  func_score
  0         1              M1E         2.0
  1         1              G3R        -7.0
  2         1              G3P        -0.5
  3         1              M1W         2.3
  4         2              M1E         1.0
  5         2              P3R        -5.0
  6         2              P3G         0.4
  7         2          M1E P3G         2.7
  8         2          M1E P3R        -2.7
  9         2              P2T         0.3


Using the predcompiled model, `global epistasis`, we can initialize the 
`Multidms` Object

>>> from multidms.model import global_epistasis
>>> mdms = multidms.Multidms(
...     func_score_df,
...     *multidms.model.global_epistasis.values(),
...     alphabet = multidms.AAS_WITHSTOP,
...     reference = "1"
... )

This object initializes a few useful attributes

>>> mdms.conditions
('1', '2')

>>> mdms.mutations
('M1E', 'M1W', 'G3P', 'G3R')

>>> mdms.site_map
   1  2
   3  G  P
   1  M  M

>>> mdms.mutations_df
  mutation         β wts  sites muts  times_seen  S_1       F_1  S_2       F_2
  0      M1E  0.080868   M      1    E           4  0.0 -0.061761  0.0 -0.061761
  1      M1W -0.386247   M      1    W           1  0.0 -0.098172  0.0 -0.098172
  2      G3P -0.375656   G      3    P           2  0.0 -0.097148  0.0 -0.097148
  3      G3R  1.668974   G      3    R           3  0.0 -0.012681  0.0 -0.012681


>>> mdms.data_to_fit
  condition aa_substitutions  ...  predicted_func_score  corrected_func_score
  0         1              G3P  ...             -0.097148                  -0.5
  1         1              G3R  ...             -0.012681                  -7.0
  2         1              M1E  ...             -0.061761                   2.0
  3         1              M1W  ...             -0.098172                   2.3
  4         2              M1E  ...             -0.089669                   1.0
  5         2          M1E P3G  ...             -0.061761                   2.7
  6         2          M1E P3R  ...             -0.011697                  -2.7
  8         2              P3G  ...             -0.066929                   0.4
  9         2              P3R  ...             -0.012681                  -5.0


We can then fit the data.

>>> data = (mdms.binarymaps['X'], mdms.binarymaps['y'])
>>> compiled_cost = global_epistasis["objective"]
>>> compiled_cost(mdms.params, data)
4.434311992312495
>>> mdms.fit()
>>> compiled_cost(mdms.params, data)
0.3332387869442089
"""
