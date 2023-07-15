"""
================================
multidms
================================

Package for modeling mutational escape from multidms antibodies using
deep mutational scanning experiments.

Importing this package imports the following objects
into the package namespace:

 - :mod:`~multidms.multidms.MultiDmsData`

 - :mod:`~multidms.multidms.MultiDmsModel`


It also imports the following alphabets:

 - :const:`~multidms.alphabets.AAS`

 - :const:`~multidms.alphabets.AAS_WITHSTOP`

 - :const:`~multidms.alphabets.AAS_WITHGAP`

 - :const:`~multidms.alphabets.AAS_WITHSTOP_WITHGAP`

"""

__author__ = "Jared Galloway"
__email__ = "jgallowa@fredhutch.org"
__version__ = "0.1.0"
__url__ = "https://github.com/matsengrp/multidms"

from polyclonal.alphabets import AAS  # noqa: F401
from polyclonal.alphabets import AAS_WITHGAP  # noqa: F401
from polyclonal.alphabets import AAS_WITHSTOP  # noqa: F401
from polyclonal.alphabets import AAS_WITHSTOP_WITHGAP  # noqa: F401

from multidms.data import MultiDmsData  # noqa: F401
from multidms.model import MultiDmsModel  # noqa: F401

import multidms.biophysical  # noqa: F401
import multidms.utils  # noqa: F401
