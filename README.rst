===============================
multidms
===============================

Model one or more deep mutational scanning (DMS) experiments
and identify variant predicted fitness, and 
individual mutation effects and shifts.

``multidms`` is a Python package written by the `Matsen Group <https://matsen.fhcrc.org/>`_ in collaboration with the `Bloom lab <https://research.fhcrc.org/bloom/en.html>`_.

The source code is `on GitHub <https://github.com/matsengrp/multidms>`_.

The documentation for this package is at `https://matsengrp.github.io/multidms/ <https://matsengrp.github.io/multidms/>`_.

To contribute to this package, read the instructions in `CONTRIBUTING.rst <CONTRIBUTING.rst>`_.

Developer install

.. code-block:: 

   git clone git@github.com:matsengrp/multidms.git
   (cd multidms && pip install -e '.[dev]')

If planning on using CUDA supported GPUs:

.. code-block:: 

   pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
