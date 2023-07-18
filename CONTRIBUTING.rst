=====================================
How to contribute to this package
=====================================

This document describes how to edit the package, run the tests, build the docs, put tagged versions on PyPI_, etc.

Editing the project
---------------------

Package structure
++++++++++++++++++
- Package code is in `multidms <multidms>`_
- Docs are in docs_
- `Jupyter notebooks`_ are in notebooks_
- Unit tests are in `tests <tests>`_

Modify the code via pull requests
+++++++++++++++++++++++++++++++++++
To make changes to the code, you should make a branch or fork, make your changes, and then submit a pull request.
If you aren't sure about pull requests:

 - A general description of pull requests: https://help.github.com/en/articles/about-pull-requests

 - How to create a pull request: https://help.github.com/en/articles/creating-a-pull-request

 - How to structure your pull requests (one conceptually distinct change per pull request): https://medium.com/@fagnerbrack/one-pull-request-one-concern-e84a27dfe9f1

 - In general, We recommend following the same instructions as `tskit <https://tskit.dev/tskit/docs/stable/development.html#sec-development-workflow-git>`_ for pull requests.


Documentation
+++++++++++++
You should document your code clearly with `numpy style documentation`_.
You may also want to write sphinx_ documentation / examples in docs_ or the notebooks_ to demonstrate large-scale functionality.

For more elaborate functionality, put unit tests in tests_.

Formatting
++++++++++
The code is formatted using `Black <https://black.readthedocs.io/en/stable/index.html>`_, which you can install using `pip install "black[jupyter]"`.
You may also wish to install a Black extension in your editor to, for example, auto-format upon save.
In any case, please run Black using `black .` before submitting your PR, because the actions tests will not pass unless the files have been formatted.
Note that this will change files/notebooks that you may be actively editing.


Adding dependencies
+++++++++++++++++++++
When you add code that uses a new package that is not in the standard python library, you should add it to the dependencies specified in the
"dependencies" section of the `pyproject.toml <pyproject.toml>`_ file.

Testing
---------

Adding tests
++++++++++++++
As you add new codes, you should create tests to make sure it is working correctly.
These can include:

  - doctests in the code

  - unit tests in the `./tests/ <tests>`_ subdirectory

Running the tests locally
++++++++++++++++++++++++++
After you make changes, you should run two sets of tests.
To run the tests, go to the top-level packag directory.
Then make sure that you have installed the packages listed in `test_requirements.txt <test_requirements.txt>`_.
If these are not installed, install them with::

    pip install -r test_requirements.txt

Then use ruff_ to `lint the code <https://en.wikipedia.org/wiki/Lint_%28software%29>`_ by running::

    ruff check .

If you need to change the ruff_ configuration, edit the `ruff.toml <ruff.toml>`_ file.

Then run the tests with pytest_ by running::

    pytest

If you need to change the pytest_ configuration, edit the `pytest.ini <pytest.ini>`_ file.

Automated testing using github actions
++++++++++++++++++++++++++++++++++++++
The aforementioned black_, ruff_ and pytest_ tests will be run automatically
by the github actions continuous integration (CI) system once a pull request is submitted.

Building documentation
------------------------
See `docs/README.rst <docs/README.rst>`_ for information on how to build the documentation.

Tagging versions and putting on PyPI
-------------------------------------
When you have a new stable release that has been commited and pushed to the main branch,
you will want to tag it and put it on PyPI_ where it can be installed with pip_.\
We reccomend making use of the bumpver_ package to do this - as the pyproject is setup to automatically fetch the latest version and
update all the version numbers in the code/docs. 

With a clean working tree, simply ::

    bumpver update --patch

Which will commit, and tag the new version. Note version format is MAJOR.MINOR.PATCH,
for major and minor version changes use the --major or --minor flags, respectively.

Next, push the new tag to the remote repository ::

    git push --tags

This will trigger the publish package action to build and upload
the package to the **test** PyPI repository.

To publish to the real PyPI_, you simply use the github web interface to pin a new release
on the tag you just created. This will trigger the publish package action to build and upload
the package to the **real** PyPI repository.

.. _pytest: https://docs.pytest.org
.. _ruff: https://github.com/charliermarsh/ruff
.. _Travis: https://docs.travis-ci.com
.. _PyPI: https://pypi.org/
.. _pip: https://pip.pypa.io
.. _sphinx: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
.. _tests: tests
.. _docs: docs
.. _notebooks: notebooks
.. _`Jupyter notebooks`: https://jupyter.org/
.. _`__init__.py`: multidms/__init__.py
.. _CHANGELOG: CHANGELOG.rst
.. _twine: https://github.com/pypa/twine
.. _`numpy style documentation`: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
.. _nbval: https://nbval.readthedocs.io
.. _bumpver: https://github.com/mbarkhau/bumpver