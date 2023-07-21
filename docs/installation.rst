Installation
============

``multidms`` requires Python 3.8 or higher.

The source code for ``multidms`` is available on GitHub at https://github.com/matsengrp/multidms.

The easiest way to install ``multidms`` is using ``pip``:

.. code-block:: 

   pip install multidms

.. note::

   The `multidms.Model` fitting process can be quite computationally intensive,
   and if available, we recommend using a GPU to accelerate it.
   While ``multidms`` ships with ``jax`` and ``jaxlib`` as dependencies,
   these packages do not include CUDA support by default.
   For this, please update the jax installation in your environment to include CUDA support
   by following the instructions in the 
   `jax documentation <https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier>`_.

Developer install
-----------------

.. code-block:: 

   git clone git@github.com:matsengrp/multidms.git
   (cd multidms && pip install -e '.[dev]')