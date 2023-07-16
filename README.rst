===============================
multidms
===============================

[![Build](https://github.com/matsengrp/multidms/actions/workflows/build_test_package.yml/badge.svg)](https://github.com/matsengrp/multidms/actions/workflows/build_test_package.yml)
[![License](https://img.shields.io/github/license/matsengrp/multidms)]
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Model one or more deep mutational scanning (DMS) experiments
and identify variant predicted fitness, and 
individual mutation effects and shifts.

``multidms`` is a Python package written by the `Matsen Group <https://matsen.fhcrc.org/>`_ in collaboration with the `Bloom lab <https://research.fhcrc.org/bloom/en.html>`_.

The source code is `on GitHub <https://github.com/matsengrp/multidms>`_.

Please see the `Documentation <https://matsengrp.github.io/multidms/>`_ for details on installation and usage.

To contribute to this package, read the instructions in `CONTRIBUTING.rst <CONTRIBUTING.rst>`_.

Developer install

.. code-block:: 

   git clone git@github.com:matsengrp/multidms.git
   (cd multidms && pip install -e '.[dev]')

If planning on using CUDA supported GPUs:

.. code-block:: 

   pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
