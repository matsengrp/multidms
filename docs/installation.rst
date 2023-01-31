Installation
============

``multidms`` requires Python 3.8 or higher.

Unfortunately ``multidms`` cannot currently be installed from `PyPI <https://pypi.org/>`_.
The reason is that ``multidms`` currently uses the `GitHub development version of altair <https://github.com/altair-viz/altair/discussions/2588>`_, which has to be installed from GitHub and `PyPI <https://pypi.org/>`_ does not allow GitHub dependencies.
Therefore, for now you have to install ``multidms`` from GitHub.
To do this for version 3.3 of ``multidms``, you would use this command::

    pip install git+https://github.com/matsengrp/multidms.git

The source code for ``multidms`` is available on GitHub at https://github.com/matsengrp/multidms.

Developer install
-----------------

.. code-block:: 

   git clone git@github.com:matsengrp/multidms.git
   (cd multidms && pip install -e '.[dev])

If planning on using CUDA supported GPU's:

.. code-block:: 

   pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
