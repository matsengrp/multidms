.. multidms documentation master file, created by
   sphinx-quickstart on Mon Jan  2 13:52:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   autodoc


``multidms`` documentation
==========================

``multidms`` is a Python package written by the 
`Matsen group <https://matsen.fhcrc.org/>`_
in collaboration with 
`William DeWitt <https://wsdewitt.github.io/>`_,
and the
`Bloom Lab <https://research.fhcrc.org/bloom/en.html>`_.
It can be used to fit a single global-epistasis model to one or more deep mutational scanning experiments, 
with the goal of estimating the effects of individual mutations, 
and how much the effects differ between experiments.

- The preprint is available on `bioRxiv <https://www.biorxiv.org/content/10.1101/2023.07.31.551037v1>`_.

- For a more advanced example of the multidms interface, see our `manuscript SARS-CoV-2 spike analysis <https://matsengrp.github.io/SARS-CoV-2_spike_multidms/spike-analysis.html>`_.

- The source code is `on GitHub <https://github.com/matsengrp/multidms>`_.

- For questions or inquiries about the software please `raise an issue <https://github.com/matsengrp/multidms/issues>`_, or contact jgallowa \<at\> fredhutch.org.

.. toctree::
    :hidden:

    self

.. toctree::
    :maxdepth: 1
    :caption: Contents
    
    installation    
    multidms
    acknowledgments
    contributing
    changelog

Simulation Analysis
-------------------

These notebooks reproduce the simulation validation from the manuscript.
Synthetic DMS data is generated with known ground-truth mutational effects and shifts,
then ``multidms`` models are fitted across a regularization grid and evaluated against the truth.
The pipeline is orchestrated by Snakemake and lives in ``experiments/simulation/``.

.. toctree::
    :maxdepth: 1
    :caption: Simulation Analysis

    sim_simulate_data
    sim_fit_models
    sim_evaluate
    sim_cross_validation
    sim_manuscript_figures

Spike Analysis
--------------

These notebooks reproduce the SARS-CoV-2 spike analysis from the manuscript.
Raw DMS data is downloaded from a public repository, processed via count
aggregation, and ``multidms`` models are fitted across a regularization grid.
The pipeline is orchestrated by Snakemake and lives in ``experiments/scv2-spike/``.

.. toctree::
    :maxdepth: 1
    :caption: Spike Analysis

    spike_prepare_data
    spike_fit_models
    spike_evaluate
    spike_cross_validation
    spike_naive_baseline
    spike_manuscript_figures

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
