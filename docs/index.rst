.. multidms documentation master file, created by
   sphinx-quickstart on Mon Jan  2 13:52:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   autodoc


``multidms`` documentation
==========================

``multidms`` is a Python package written by the `Matsen <https://matsen.fhcrc.org/>`_ and `Bloom <https://research.fhcrc.org/bloom/en.html>`_ labs.
It can be used to fit a single global-epistasis model to one or more deep mutational scanning experiments, 
with the goal of estimating the effects of individual mutations, 
and how much the effects differ between experiments.

The source code is `on GitHub <https://github.com/matsengrp/multidms>`_.

See below for information and examples of how to use this package.

.. toctree::
    :hidden:

    self

.. toctree::
    :maxdepth: 2
    :caption: Contents
    
    installation    
    biophysical_model
    fit_delta_BA1_example
    autodoc
    acknowledgments
..   
    jit model composition
    using with GPU's
    contributing


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
