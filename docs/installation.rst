Installation
============

``multidms`` requires Python 3.8 or higher.

The source code for ``multidms`` is available on GitHub at https://github.com/matsengrp/multidms.

Developer install
-----------------

.. code-block:: 

   git clone git@github.com:matsengrp/multidms.git
   (cd multidms && pip install -e '.[dev]')

If planning on using CUDA supported GPU's:

.. code-block:: 

   pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
