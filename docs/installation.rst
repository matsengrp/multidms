Installation
============

``multidms`` requires Python 3.9 or higher.

The source code is available on GitHub at
https://github.com/matsengrp/multidms.

Install from PyPI
-----------------

.. code-block::

   pip install multidms

Developer install
-----------------

Option A: pixi (recommended)
+++++++++++++++++++++++++++++

`pixi <https://pixi.sh>`_ provides declarative, one-command environment
setup with pinned Python versions.

Install pixi (`instructions <https://pixi.sh/latest/#installation>`_),
then::

    pixi install          # creates env, installs all deps + editable package
    pixi run test         # pytest with doctests
    pixi run lint         # ruff
    pixi run fmt          # black
    pixi run docs         # build Sphinx docs

To test against a specific Python version::

    pixi run -e py39 test
    pixi run -e py312 test

Option B: pip
+++++++++++++

::

    git clone git@github.com:matsengrp/multidms.git
    cd multidms
    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest --doctest-modules multidms tests
    ruff check .
    black .

Both approaches are fully supported.
See `CONTRIBUTING.rst <https://github.com/matsengrp/multidms/blob/main/CONTRIBUTING.rst>`_
for contribution guidelines.
