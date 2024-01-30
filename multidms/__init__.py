"""
========
multidms
========

multidms is a Python package for modeling deep mutational scanning data.
In particular, it is designed to model data from more than one experiment,
even if they don't share the same wildtype amino acid sequence.
It uses joint modeling to inform parameters across all experiments,
while identifying experiment-specific mutation effects which differ.

.. currentmodule:: multidms


Importing this package imports the following objects
into the package namespace:

 - :mod:`~multidms.data.Data`

 - :mod:`~multidms.model.Model`
 
 - :mod:`~multidms.model_collection.ModelCollection`

For a brief description about how the :class:`~multidms.model.Model`
class works to compose, compile, and optimize the model parameters
- as well as detailed code code documentation for each of the
equations described in the 
`biophysical docs <https://matsengrp.github.io/multidms/biophysical_model.html>`_ - 
see:

 - :mod:`~multidms.biophysical`

:mod:`~multidms.plot` mostly contains code for interactive plotting
at the moment.

It also imports the following alphabets:

 - :const:`~multidms.alphabets.AAS`

 - :const:`~multidms.alphabets.AAS_WITHSTOP`

 - :const:`~multidms.alphabets.AAS_WITHGAP`

 - :const:`~multidms.alphabets.AAS_WITHSTOP_WITHGAP`

"""

__author__ = "Jared Galloway"
__email__ = "jgallowa@fredhutch.org"
__version__ = "0.3.3"
__url__ = "https://github.com/matsengrp/multidms"

from polyclonal.alphabets import AAS  # noqa: F401
from polyclonal.alphabets import AAS_WITHGAP  # noqa: F401
from polyclonal.alphabets import AAS_WITHSTOP  # noqa: F401
from polyclonal.alphabets import AAS_WITHSTOP_WITHGAP  # noqa: F401

from multidms.data import Data  # noqa: F401
from multidms.model import Model  # noqa: F401
from multidms.model_collection import ModelCollection, fit_models  # noqa: F401

# This lets Sphinx know you want to document foo.foo.Foo as foo.Foo.
__all__ = ["Data", "Model", "ModelCollection", "fit_models"]
