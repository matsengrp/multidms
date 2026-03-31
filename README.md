# multidms

![License](https://img.shields.io/github/license/matsengrp/multidms)
[![PyPI version](https://badge.fury.io/py/multidms.svg)](https://badge.fury.io/py/multidms)
[![Build](https://github.com/matsengrp/multidms/actions/workflows/build_test_package.yml/badge.svg)](https://github.com/matsengrp/multidms/actions/workflows/build_test_package.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

`multidms` is a Python package written by the 
[Matsen group](https://matsen.fhcrc.org/)
in collaboration with 
[William DeWitt](https://wsdewitt.github.io/)
and the
[Bloom Lab](https://research.fhcrc.org/bloom/en.html).
It can be used to jointly fit a global-epistasis model to one or more deep mutational scanning experiments, 
with the goal of estimating the effects of individual mutations, 
and how much the effects differ between experiments.

The source code is [on GitHub](https://github.com/matsengrp/multidms).

Please see the [Documentation](https://matsengrp.github.io/multidms/) for details on installation and usage.

To contribute to this package, read the instructions in [CONTRIBUTING.rst](CONTRIBUTING.rst).

## Development

### Option A: pixi (recommended)

[pixi](https://pixi.sh) provides declarative, one-command environment setup with pinned Python versions.

Install pixi ([instructions](https://pixi.sh/latest/#installation)), then:

    pixi install          # creates env, installs all deps + editable package
    pixi run test         # pytest with doctests
    pixi run lint         # ruff
    pixi run fmt          # black
    pixi run docs         # build Sphinx docs

To test against a specific Python version:

    pixi run -e py39 test
    pixi run -e py312 test

### Option B: pip

    python -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest --doctest-modules multidms tests
    ruff check .
    black .

Both approaches are fully supported. See [CONTRIBUTING.rst](CONTRIBUTING.rst) for more details.