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

# TODO

# TODO __author__ = "`the Matsen & Bloom labs <https://research.fhcrc.org/bloom/en.html>`_"
__email__ = "jgallowa@fredhutch.org"
__version__ = "0.0.1"
__url__ = "https://github.com/matsengrp/multidms"

from polyclonal.alphabets import AAS
from polyclonal.alphabets import AAS_WITHGAP
from polyclonal.alphabets import AAS_WITHSTOP
from polyclonal.alphabets import AAS_WITHSTOP_WITHGAP

from multidms.data import MultiDmsData
from multidms.model import MultiDmsModel

import multidms.biophysical
import multidms.utils

# import multidms.model
# import multidms.utils
