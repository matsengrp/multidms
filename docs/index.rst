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

- A concise description of the joint modeling approach is available in the `biophysical model <https://matsengrp.github.io/multidms/biophysical_model.html>`_ section.

- A example fitting python with the python interface is available in the `usage examples documentation <https://matsengrp.github.io/multidms/fit_delta_BA1_example.html>`_ page.

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
    biophysical_model
    simulation_validation
    fit_delta_BA1_example
    multidms
    acknowledgments
    contributing
    changelog


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
