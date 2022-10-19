"""
==========
multidms
==========

Defines :class:`Multidms` objects for handling data from one or more
dms experiments under various conditions.

"""


import collections
import copy  # noqa: F401
import inspect
import itertools
import os
import sys
import time
import json

import binarymap
from polyclonal.utils import MutationParser
import frozendict
# import natsort
import numpy as onp
import pandas

#import scipy.optimize
#import scipy.special
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental import sparse

import multidms
import multidms.plot
import multidms.utils


class MultidmsFitError(Exception):
    """Error fitting in :meth:`Multidms.fit`."""

    pass


class MultidmsHarmonizeError(Exception):
    """Error harmonizing conditions in :meth:`Multidms.condition_harmonized_model`."""

    pass


class Multidms:
    r"""Represent and model one or more dms experiments
    and identify variant predicted fitness, and 
    individual mutation effects and shifts.

    Note
    ----
    
    multidms
    ========
    
    Overview of model
    -----------------
    
    The ``multidms`` model applies to a case where you have DMS datasets 
    for two or more conditions and are interested in identifying shifts 
    in mutational effects between conditions.
    To do so, the model defines one condition as a reference condition.
    For each mutation, the model fits one parameter that quantifies 
    the effect of the mutation in the reference condition.
    For each non-reference condition, it also fits a shift 
    parameter that quantifies the shift in the mutation's 
    effect in the non-reference condition relative to the reference.
    Shift parameters can be regularized, encouraging most of them to be 
    close to zero. This regularization step is a useful way to eliminate 
    the effects of experimental noise, and is most useful in cases where 
    you expect most mutations to have the same effects between conditions, 
    such as for conditions that are close relatives.
    
    The model uses a global-epistasis function to disentangle the effects 
    of multiple mutations on the same variant. To do so, it assumes 
    that mutational effects additively influence a latent biophysical 
    property the protein (e.g., $\Delta G$ of folding).
    The mutational-effect parameters described above operate at this latent level.
    
    The global-epistasis function then assumes a sigmoidal relationship between 
    a protein's latent property and its functional score measured in the experiment 
    (e.g., log enrichment score). Ultimately, mutational parameters, as well as ones 
    controlling the shape of the sigmoid, are all jointly fit to maximize agreement 
    between predicted and observed functional scores acorss all variants of all conditions.
    
    Detailed description of the model
    ---------------------------------
    
    For each variant $v$ from condition $h$, we use a global-epistasis function 
    $g$ to convert a latent phenotype $\phi$ to a functional score $f$:
    
    $$f(v,h) = g_{\alpha}(\phi(v,h)) + γ_h$$
    
    where $g$ is a sigmoid and $\alpha$ is a set of parameters,
    ``ge_scale``\ , and ``ge_bias`` which define the shape of the sigmoid.
    
    The latent phenotype is computed in the following way:
    
    $$\phi(v,h) = c + \sum_{m \in v} (x\ *m + s*\ {m,h})$$
    
    where:
    
    
    * $c$ is the wildtype latent phenotype for the reference condition.
    * $x_m$ is the latent phenotypic effect of mutation $m$. See details below.
    * $s_{m,h}$ is the shift of the effect of mutation $m$ in condition $h$. 
      These parameters are fixed to zero for the reference condition. For 
      non-reference conditions, they are defined in the same way as $x_m$ parameters.
    * $v$ is the set of all mutations relative to the reference wildtype sequence 
      (including all mutations that separate condition $h$ from the reference condition).
    
    The $x_m$ variable is defined such that mutations are always relative to the 
    reference condition. For example, if the wildtype amino acid at site 30 is an 
    A in the reference condition, and a G in a non-reference condition, 
    then a Y30G mutation in the non-reference condition is recorded as an A30G 
    mutation relative to the reference. This way, each condition informs 
    the exact same parameters, even at sites that differ in wildtype amino acid.
    These are encoded in a ``BinaryMap`` object, where all sites that are non-identical 
    to the reference are 1's.
    
    Ultimately, we fit parameters using a loss function with one term that 
    scores differences between predicted and observed values and another that 
    uses L1 regularization to penalize non-zero $s_{m,h}$ values:
    
    $$ L\ *{\text{total}} = \sum*\ {h} \left[\sum\ *{v} L*\ {\text{fit}}(y\ *{v,h}, f(v,h)) + \lambda \sum*\ {m} |s_{m,h}|\right]$$
    
    where:
    
    
    * $L_{\text{total}}$ is the total loss function.
    * $L_{\text{fit}}$ is a loss function that penalizes differences 
        in predicted vs. observed functional scores.
    * $y_{v,h}$ is the experimentally measured functional score of 
        variant $v$ from condition $h$.
    
    Model using matrix algebra
    --------------------------
    
    We compute a vector or predicted latent phenotypes $P_{h}$ as:
    
    $$P_{h} = c + (X_h \cdot (β + S_h))$$
    
    where:
    
    
    * $β$ is a vector of all $β_m$ values.
    * $S\ *h$ is a matrix of all $s*\ {m,h}$ values.
    * $X_h$ is a sparse matrix, where rows are variants, 
        columns are mutations (all defined relative to the reference condition), 
        and values are weights of 0's and 1's. These weights are used to 
        compute the phenotype of each variant given the mutations present.
    * $c$ is the same as above.
    
    In the matrix algebra, the sum of $β\ *m$ and $S*\ {m,h}$ gives a vector of mutational effects, with one entry per mutation.
    Multiplying the matrix $X_h$ by this vector gives a new vector with one entry per variant, where values are the sum of mutational effects, weighted by the variant-specific weights in $X_h$.
    Adding the $c$ value to this vector will give a vector of predicted latent phenotypes for each variant.
    
    Next, the global-epistasis function can be used to convert a vector of predicted latent phenotypes to a vector of predicted functional scores.
    
    $$F\ *{h,pred} = g*\ {\alpha}(P_h)$$
    
    Finally, this vector could be fed into a loss function and compared with a vector of observed functional scores.

    Note
    ----
    You can initialize a :class:`Multidms` object in two (?) ways:

    1. functional dataframe
        TODO
        With data to fit the condition activities and mutation-shift values,
        and initial guesses for the condition activities and mutation-shift
        values. To do this, initialize with ``data_to_fit`` holding the data,
        ``activity_wt_df`` holding initial guesses of activities, and
        ``mut_shift_df`` or ``site_escape_df`` holding initial guesses for
        mutation shifts (see also ``init_missing`` and
        ``data_mut_shift_overlap``). Then call :meth:`Multidms.fit`.

    Parameters
    ----------
    data_to_fit : pandas.DataFrame or None
        TODO
        Should have columns named 'aa_substitutions', 'concentration', and
        'prob_shift'. The 'aa_substitutions' column defines each variant
        :math:`v` as a string of substitutions (e.g., 'M3A K5G').
    reference : str
        Name of the factor level which annotates the reference condition
        variants. See the class Note above.
    init_parameters : pandas.DataFrame or None
        TODO
        Should have columns named 'mutation', 'condition', and 'shift' that
        give the :math:`\beta_{m,e}` values (in the 'shift' column), with
        mutations written like "G7M".
    collapse_identical_variants : {'mean', 'median', False}
        TODO
        If identical variants in ``data_to_fit`` (same 'aa_substitutions'),
        collapse them and make weight proportional to number of collapsed
        variants? Collapse by taking mean or median of 'prob_shift', or
        (if `False`) do not collapse at all. Collapsing will make fitting faster,
        but *not* a good idea if you are doing bootstrapping.
    alphabet : array-like
        Allowed characters in mutation strings.
    sites : array-like or 'infer'
        TODO - this is good idea if we want to define the site map for the reference
        By default, sites are assumed to be sequential integer values are and inferred
        from ``data_to_fit`` or ``mut_shift_df``. However, you can also have
        non-sequential integer sites, or sites with lower-case letter suffixes
        (eg, `214a`) if your protein is numbered against a reference that it has
        indels relative to. In that case, provide list of all expected in order
        here; we require that order to be natsorted.
    condition_colors : array-like or dict
        Maps each condition to the color used for plotting. Either a dict keyed
        by each condition, or an array of colors that are sequentially assigned
        to the conditions.
    init_missing : 'zero' or int
        How to initialize activities or mutation-shift values not specified in
        ``activity_wt_df`` or ``mut_shift_df`` / ``site_escape_df``. If
        'zero', set mutation-shifts to zero and activities uniformly spaced
        from 1 to 0. Otherwise draw uniformly from between 0 and 1 using
        specified random number seed.
    condition_mut_shift_overlap : str: "sites", "mutations", None
        How should the conditional data be curated for fitting?


    Attributes
    ----------
    conditions : tuple
        Names of all conditions.

    mutations : pandas.DataFrame
        A dataframe containing all relevant substitutions and all relevant.
    mutations_times_seen : frozendict.frozendict or None
        If `data_to_fit` is not `None`, keyed by all mutations with shift values
        and values are number of variants in which the mutation is seen. It is formally
        calculated as the number of variants with mutation across all concentrations
        divided by the number of concentrations, so can have non-integer values if
        there are variants only observed at some concentrations.
        1. mutation_shifts : TODO
        2. mutation_betas : TODO
        3. mutation_functional_effects : TODO
    reference : TODO
    alphabet : tuple 
        Allowed characters in mutation strings.
    sites : tuple TODO 
        List of all sites. These are the sites provided via the ``sites`` parameter,
        or inferred from ``data_to_fit`` or ``mut_shift_df`` if that isn't provided.
        If `sequential_integer_sites` is `False`, these are str, otherwise int.
    sequential_integer_sites : bool TODO
        True if sites are sequential and integer, False otherwise.
    condition_colors : dict TODO
        Maps each condition to its color.
    data_to_fit : pandas.DataFrame or None TODO 
        Data to fit as passed when initializing this :class:`Multidms` object.
        If using ``collapse_identical_variants``, then identical variants
        are collapsed on columns 'concentration', 'aa_substitutions',
        and 'prob_shift', and a column 'weight' is added to represent number
        of collapsed variants. Also, row-order may be changed.

    Example
    -------
    # TODO make example - check notebook
    Simple example with two conditions (`e1` and `e2`) and a few mutations where
    we know the activities and mutation-level shift values ahead of time:

    >>> activity_wt_df = pandas.DataFrame({'condition':  ['e1', 'e2'],
    ...                                'activity': [ 2.0,  1.0]})
    >>> func_scores_df = pandas.DataFrame({
    ...   'aa_substitutions': ['M1C', 'M1C', 'G2A', 'G2A', 'A4K', 'A4K', 'A4L', 'A4L'],
    ...   'condition':  [ 'e1',  'e2',  'e1',  'e2',  'e1',  'e2',  'e1',  'e2'],
    ...   })
    >>> model = Multidms(
    ...     activity_wt_df=activity_wt_df, #TODO or "infer"
    ...     mut_shift_df=func_score_df,
    ...     collapse_identical_variants="mean",
    ... )
    >>> model.conditions
    TODO
    ('e1', 'e2')
    >>> model.mutations
    TODO
    ('M1C', 'G2A', 'A4K', 'A4L')
    >>> model.mutations_times_seen is None
    TODO
    True
    >>> model.sites_df
    TODO
    >>> model.activity_wt_df
    TODO
      condition  activity
    0      e1       2.0
    1      e2       1.0
    >>> model.mutations_df
    TODO 
      condition  site wildtype mutant mutation  shift
    0      e1     1        M      C      M1C     2.0
    1      e1     2        G      A      G2A     3.0
    2      e1     4        A      K      A4K     0.0
    3      e1     4        A      L      A4L     0.0
    4      e2     1        M      C      M1C     0.0
    5      e2     2        G      A      G2A     0.0
    6      e2     4        A      K      A4K     2.5
    7      e2     4        A      L      A4L     1.5

    We can also summarize the mutation-level shift at the site level:

    >>> pandas.set_option("display.max_columns", None)
    >>> pandas.set_option("display.width", 89)
    >>> model.mut_shift_site_summary_df()
    TODO
      condition  site wildtype  mean  total positive  max  min  total negative  n mutations
    0      e1     1        M   2.0             2.0  2.0  2.0             0.0            1
    1      e1     2        G   3.0             3.0  3.0  3.0             0.0            1
    2      e1     4        A   0.0             0.0  0.0  0.0             0.0            2
    3      e2     1        M   0.0             0.0  0.0  0.0             0.0            1
    4      e2     2        G   0.0             0.0  0.0  0.0             0.0            1
    5      e2     4        A   2.0             4.0  2.5  1.5             0.0            2

    Note that we can **not** initialize a :class:`Multidms` object if we are
    missing shift estimates for any mutations for any conditions:

    >>> Multidms(activity_wt_df=activity_wt_df,
    ...            mut_shift_df=mut_escape_df.head(n=5))
    Traceback (most recent call last):
      ...
    ValueError: invalid set of mutations for condition='e2'

    Now make a data frame with some variants:

    >>> variants_df = pandas.DataFrame.from_records(
    ...         [('AA', ''),
    ...          ('AC', 'M1C'),
    ...          ('AG', 'G2A'),
    ...          ('AT', 'A4K'),
    ...          ('TA', 'A4L'),
    ...          ('CA', 'M1C G2A'),
    ...          ('CG', 'M1C A4K'),
    ...          ('CC', 'G2A A4K'),
    ...          ('TC', 'G2A A4L'),
    ...          ('CT', 'M1C G2A A4K'),
    ...          ('TG', 'M1C G2A A4L'),
    ...          ('GA', 'M1C'),
    ...          ],
    ...         columns=['barcode', 'aa_substitutions'])

    Get the shift probabilities predicted on these variants from
    the values in the :class:`Multidms` object:

    >>> shift_probs = model.prob_escape(variants_df=variants_df,
    ...                                  concentrations=[1.0, 2.0, 4.0])
    >>> shift_probs.round(3)
       barcode aa_substitutions  concentration  predicted_prob_shift
    0       AA                             1.0                  0.032
    1       AT              A4K            1.0                  0.097
    2       TA              A4L            1.0                  0.074
    3       AG              G2A            1.0                  0.197
    4       CC          G2A A4K            1.0                  0.598
    5       TC          G2A A4L            1.0                  0.455
    6       AC              M1C            1.0                  0.134
    7       GA              M1C            1.0                  0.134
    8       CG          M1C A4K            1.0                  0.409
    9       CA          M1C G2A            1.0                  0.256
    10      CT      M1C G2A A4K            1.0                  0.779
    11      TG      M1C G2A A4L            1.0                  0.593
    12      AA                             2.0                  0.010
    13      AT              A4K            2.0                  0.044
    14      TA              A4L            2.0                  0.029
    15      AG              G2A            2.0                  0.090
    16      CC          G2A A4K            2.0                  0.398
    17      TC          G2A A4L            2.0                  0.260
    18      AC              M1C            2.0                  0.052
    19      GA              M1C            2.0                  0.052
    20      CG          M1C A4K            2.0                  0.230
    21      CA          M1C G2A            2.0                  0.141
    22      CT      M1C G2A A4K            2.0                  0.629
    23      TG      M1C G2A A4L            2.0                  0.411
    24      AA                             4.0                  0.003
    25      AT              A4K            4.0                  0.017
    26      TA              A4L            4.0                  0.010
    27      AG              G2A            4.0                  0.034
    28      CC          G2A A4K            4.0                  0.214
    29      TC          G2A A4L            4.0                  0.118
    30      AC              M1C            4.0                  0.017
    31      GA              M1C            4.0                  0.017
    32      CG          M1C A4K            4.0                  0.106
    33      CA          M1C G2A            4.0                  0.070
    34      CT      M1C G2A A4K            4.0                  0.441
    35      TG      M1C G2A A4L            4.0                  0.243

    We can also get predicted shift probabilities by including concentrations
    in the data frame passed to :meth:`Multidms.prob_shift`:

    >>> model.prob_shift(
    ...         variants_df=pandas.concat([variants_df.assign(concentration=c)
    ...                                for c in [1.0, 2.0, 4.0]])
    ...         ).equals(shift_probs)
    True

    We can also compute the IC50s:

    >>> model.icXX(variants_df).round(3)
       barcode aa_substitutions   IC50
    0       AA                   0.085
    1       AC              M1C  0.230
    2       GA              M1C  0.230
    3       AG              G2A  0.296
    4       AT              A4K  0.128
    5       TA              A4L  0.117
    6       CA          M1C G2A  0.355
    7       CG          M1C A4K  0.722
    8       CC          G2A A4K  1.414
    9       TC          G2A A4L  0.858
    10      CT      M1C G2A A4K  3.237
    11      TG      M1C G2A A4L  1.430

    Or the IC90s:

    >>> model.icXX(variants_df, x=0.9, col='IC90').round(3)
       barcode aa_substitutions    IC90
    0       AA                    0.464
    1       AC              M1C   1.260
    2       GA              M1C   1.260
    3       AG              G2A   1.831
    4       AT              A4K   0.976
    5       TA              A4L   0.782
    6       CA          M1C G2A   2.853
    7       CG          M1C A4K   4.176
    8       CC          G2A A4K   7.473
    9       TC          G2A A4L   4.532
    10      CT      M1C G2A A4K  18.717
    11      TG      M1C G2A A4L   9.532

    >>> model_data.mutations
    ('M1C', 'G2A', 'A4K', 'A4L')
    >>> dict(model_data.mutations_times_seen)
    {'G2A': 6, 'M1C': 6, 'A4K': 4, 'A4L': 3}

    The activities are evenly spaced from 1 to 0, while the mutation shifts
    are all initialized to zero:

    >>> model_data.condition_df
      condition  activity
    0       1       1.0
    1       2       0.0
    >>> model_data.mut_df
      condition  site wildtype mutant mutation  shift  times_seen
    0       1     1        M      C      M1C     0.0           6
    1       1     2        G      A      G2A     0.0           6
    2       1     4        A      K      A4K     0.0           4
    3       1     4        A      L      A4L     0.0           3
    4       2     1        M      C      M1C     0.0           6
    5       2     2        G      A      G2A     0.0           6
    6       2     4        A      K      A4K     0.0           4
    7       2     4        A      L      A4L     0.0           3

    You can initialize to random numbers by setting ``init_missing`` to seed
    (in this example we also don't include all variants for one concentration):

    >>> model_data2 = Multidms(
    ...     data_to_fit=data_to_fit.head(30),
    ...     n_conditions=2,
    ...     init_missing=1,
    ...     collapse_identical_variants="mean",
    ... )
    >>> model_data2.activity_wt_df.round(3)
      condition  activity
    0       1     0.417
    1       2     0.720

    You can set some or all mutation shifts to initial values:

    >>> model_data3 = Multidms(
    ...     data_to_fit=data_to_fit,
    ...     activity_wt_df=activity_wt_df,
    ...     mut_shift_df=pandas.DataFrame({'condition': ['e1'],
    ...                                 'mutation': ['M1C'],
    ...                                 'shift': [4]}),
    ...     data_mut_shift_overlap='fill_to_data',
    ...     collapse_identical_variants="mean",
    ... )
    >>> model_data3.mut_shift_df
      condition  site wildtype mutant mutation  shift  times_seen
    0      e1     1        M      C      M1C     4.0           6
    1      e1     2        G      A      G2A     0.0           6
    2      e1     4        A      K      A4K     0.0           4
    3      e1     4        A      L      A4L     0.0           3
    4      e2     1        M      C      M1C     0.0           6
    5      e2     2        G      A      G2A     0.0           6
    6      e2     4        A      K      A4K     0.0           4
    7      e2     4        A      L      A4L     0.0           3

    You can initialize **sites** to shift values via ``site_activity_df``:

    >>> model_data4 = Multidms(
    ...     data_to_fit=data_to_fit,
    ...     activity_wt_df=activity_wt_df,
    ...     site_shift_df=pandas.DataFrame.from_records(
    ...         [('e1', 1, 1.0), ('e1', 4, 0.0),
    ...          ('e2', 1, 0.0), ('e2', 4, 2.0)],
    ...         columns=['condition', 'site', 'shift'],
    ...     ),
    ...     data_mut_shift_overlap='fill_to_data',
    ...     collapse_identical_variants="mean",
    ... )
    >>> model_data4.mut_shift_df
      condition  site wildtype mutant mutation  shift  times_seen
    0      e1     1        M      C      M1C     1.0           6
    1      e1     2        G      A      G2A     0.0           6
    2      e1     4        A      K      A4K     0.0           4
    3      e1     4        A      L      A4L     0.0           3
    4      e2     1        M      C      M1C     0.0           6
    5      e2     2        G      A      G2A     0.0           6
    6      e2     4        A      K      A4K     2.0           4
    7      e2     4        A      L      A4L     2.0           3

    Fit the data using :meth:`Multidms.fit`, and make sure the new
    predicted shift probabilities are close to the real ones being fit.
    Reduce weight on regularization since there is so little data in this
    toy example:

    >>> for m in [model_data, model_data2, model_data3, model_data4]:
    ...     opt_res = m.fit(
    ...         reg_shift_weight=0.001,
    ...         reg_spread_weight=0.001,
    ...         reg_activity_weight=0.0001,
    ...     )
    ...     pred_df = m.prob_shift(variants_df=data_to_fit)
    ...     if not numpy.allclose(pred_df['prob_shift'],
    ...                           pred_df['predicted_prob_shift'],
    ...                           atol=0.01):
    ...          raise ValueError(f"wrong predictions\n{pred_df}")
    ...     if not numpy.allclose(
    ...              activity_wt_df['activity'].sort_values(),
    ...              m.activity_wt_df['activity'].sort_values(),
    ...              atol=0.1,
    ...              ):
    ...          raise ValueError(f"wrong activities\n{m.activity_wt_df}")
    ...     if not numpy.allclose(
    ...              mut_shift_df['escape'].sort_values(),
    ...              m.mut_shift_df['escape'].sort_values(),
    ...              atol=0.05,
    ...              ):
    ...          raise ValueError(f"wrong shifts\n{m.mut_escape_df}")

    >>> model_data.mut_shift_site_summary_df().round(1)
      condition  site wildtype  mean  total positive  max  min  total negative  n mutations
    0       1     1        M   0.0             0.0  0.0  0.0             0.0            1
    1       1     2        G   0.0             0.0  0.0  0.0             0.0            1
    2       1     4        A   2.0             4.0  2.5  1.5             0.0            2
    3       2     1        M   2.0             2.0  2.0  2.0             0.0            1
    4       2     2        G   3.0             3.0  3.0  3.0             0.0            1
    5       2     4        A   0.0             0.0  0.0  0.0             0.0            2
    >>> model_data.mut_shift_site_summary_df(min_times_seen=4).round(1)
      condition  site wildtype  mean  total positive  max  min  total negative  n mutations
    0       1     1        M   0.0             0.0  0.0  0.0             0.0            1
    1       1     2        G   0.0             0.0  0.0  0.0             0.0            1
    2       1     4        A   2.5             2.5  2.5  2.5             0.0            1
    3       2     1        M   2.0             2.0  2.0  2.0             0.0            1
    4       2     2        G   3.0             3.0  3.0  3.0             0.0            1
    5       2     4        A   0.0             0.0  0.0  0.0             0.0            1


    TODO
    Example
    -------
    Filter variants by how often they are seen in data:

    >>> model_data.filter_variants_by_seen_muts(variants_df)
    ... # doctest: +NORMALIZE_WHITESPACE
       barcode aa_substitutions
    0       AA
    1       AC              M1C
    2       AG              G2A
    3       AT              A4K
    4       TA              A4L
    5       CA          M1C G2A
    6       CG          M1C A4K
    7       CC          G2A A4K
    8       TC          G2A A4L
    9       CT      M1C G2A A4K
    10      TG      M1C G2A A4L
    11      GA              M1C

    >>> model_data.filter_variants_by_seen_muts(variants_df, min_times_seen=5)
    ... # doctest: +NORMALIZE_WHITESPACE
      barcode aa_substitutions
    0      AA
    1      AC              M1C
    2      AG              G2A
    3      CA          M1C G2A
    4      GA              M1C

    >>> model_data.filter_variants_by_seen_muts(variants_df, min_times_seen=4)
    ... # doctest: +NORMALIZE_WHITESPACE
      barcode aa_substitutions
    0      AA
    1      AC              M1C
    2      AG              G2A
    3      AT              A4K
    4      CA          M1C G2A
    5      CG          M1C A4K
    6      CC          G2A A4K
    7      CT      M1C G2A A4K
    8      GA              M1C


    """

    # TODO init_parameters : pandas.DataFrame or None
    # Offer ability ot set parameters yourself?
    # How would you eventually understand what 
    # the parameters shapes should be?

    # TODO sites : array-like or 'infer' #2
    # Offer ability to provide your own refernce sites?
    # Right now the inference seems like the best way to go?
    # except, it seems to me that we always want these models to
    # to be sparse unless we want to predict?

    # TODO *, ? What's with this parameter? I'm assuming it's for
    # child classes.

    # TODO condition_colors : array-like or dict
    # Offer option to provide condition colors, or cmap / palette?

    # kwargs
    # condition_mut_shift_overlap="exact_match",
    # collapse_identical_variants : str or None,


    def __init__(
        self,
        data_to_fit : pandas.DataFrame,
        reference : str,
        alphabet=multidms.AAS,
        init_missing="zero",
        **kwargs
    ):
        """See main class docstring."""

        # check init seed 
        # TODO Is this necessary?
        if isinstance(init_missing, int):
            # TODO set the JAX PRNG key?
            numpy.random.seed(init_missing)
        elif init_missing != "zero":
            raise ValueError(f"invalid {init_missing=}")    

        # Check and initialize condition colors
        if isinstance(condition_colors, dict):
            self.condition_colors = {e: condition_colors[e] for e in self.conditions}
        elif len(condition_colors) < len(self.conditions):
            raise ValueError("not enough `condition_colors`")
        else:
            self.condition_colors = dict(zip(self.conditions, condition_colors))
        
        # Check and initialize alphabet & mut parser attributes
        if len(set(alphabet)) != len(alphabet):
            raise ValueError("duplicate letters in `alphabet`")
        self.alphabet = tuple(alphabet)
        self._mutparser = MutationParser(
            alphabet,
            letter_suffixed_sites=not self.sequential_integer_sites,
        )

        # Check and initialize conditions attribute
        if pandas.isnull(activity_wt_df["condition"]).any():
            raise ValueError("condition name cannot be null")
        self.conditions = tuple(activity_wt_df["condition"].unique())

        # Check and initialize fitting data from func_score_df
        (
            self._binarymaps,
            self._data_to_fit,
            self._all_subs,
            self._site_map,
        ) = self._create_condition_modeling_data(
            data_to_fit, 
            reference,
            collapse_identical_variants,
            **kwargs    
        )

        # set internal params with activities and shifts
        self._params = self._initialize_model_params()

        # TODO initialize the mutations_df

        """
        def _init_mut_shift_df(mutations):
            # initialize mutation shift values
            if init_missing == "zero":
                init = 0.0
            else:
                init = numpy.random.rand(len(self.conditions) * len(mutations))
            return pandas.DataFrame(
                {
                    "condition": list(self.conditions) * len(mutations),
                    "mutation": [m for m in mutations for _ in self.conditions],
                    "shift": init,
                }
            )
        """

        # TODO Compute times seen somewhere

        """
        # get wildtype, sites, and mutations
        if data_to_fit is not None:
            wts2, sites2, muts2 = self._muts_from_data_to_fit(data_to_fit)
            if (self.sites is not None) and not set(sites2).issubset(self.sites):
                raise ValueError("sites in `data_to_fit` not all in `sites`")
            times_seen = (
                data_to_fit["aa_substitutions"]
                .str.split()
                .explode()
                .dropna()
                .value_counts()
                .sort_values(ascending=False)
                / data_to_fit["concentration"].nunique()
            )
            if (times_seen == times_seen.astype(int)).all():
                times_seen = times_seen.astype(int)
            self.mutations_times_seen = frozendict.frozendict(times_seen)
        """

    # TODO impliment different condition overlap options
    def _create_condition_modeling_data(
        func_score_df:pandas.DataFrame,
        reference_condition:str,
        condition_overlap="sites",
        collapse_identical_variants=False,
        condition_col="condition",
        substitution_col="aa_substitutions",
        func_score_col="func_score",
        **kwargs
    ):
        """
        Takes a dataframe for making a `BinaryMap` object, and adds
        a column where each entry is a list of mutations in a variant
        relative to the amino-acid sequence of the reference condition.
        
        Parameters
        ----------
    
        func_score_df : pandas.DataFrame
            This should be in the same format as described in BinaryMap.

        collapse_identical_variants : {'mean', 'median', False}
            If identical variants in ``data_to_fit`` (same 'aa_substitutions'),
            collapse them and make weight proportional to number of collapsed
            variants? Collapse by taking mean or median of 'prob_escape', or
            (if `False`) do not collapse at all. Collapsing will make fitting faster,
            but *not* a good idea if you are doing bootstrapping.
        
        condition_col : str
            The name of the column in func_score_df that identifies the
            condition for a given variant. We require that the
            reference condition variants are labeled as 'reference'
            in this column.
            
        reference_condition : str
            The name of the condition existing in ``condition_col`` for
            which we should convert all substitution to be with respect to.
        
        substitution_col : str 
            The name of the column in func_score_df that
            lists mutations in each variant relative to the condition wildtype
            amino-acid sequence where sites numbers must come from an alignment
            to a reference sequence (which may or may not be the same as the
            reference condition).
            
        func_score_col : str
            Column in func_scores_df giving functional score for each variant.
            
        
        Returns
        -------
            
        tuple : (dict[BinaryMap], dict[jnp.array]), pandas.DataFrame, np.array, pd.DataFrame
        
            This function return a tuple which can be unpacked into the following:
            
            - (X, y) Where X and y are both dictionaries containing the prepped data
                for training our JAX multidms model. The dictionary keys
                stratify the datasets by condition
                
            - A pandas dataframe which primary contains the information from
                func_score_df, but has been curated to include only the variants
                deemed appropriate for training, as well as the substitutions
                converted to be wrt to the reference condition.
                
            - A numpy array giving the substitutions (beta's) of the binary maps
                in the order that is preserved to match the matrices in X.
                
            - A pandas dataframe providing the site map indexed by alignment site to
                a column for each condition wt amino acid. 
        
        """

        cols = ["concentration", "aa_substitutions"]
        if "weight" in df.columns:
            cols.append(
                "weight"
            )  # will be overwritten if `collapse_identical_variants`
        #if get_pv:
        #    cols.append("prob_shift")
        if not df[cols].notnull().all().all():
            raise ValueError(f"null entries in data frame of variants:\n{df[cols]}")

        if collapse_identical_variants:
            agg_dict = {"weight": "sum"}
            if get_pv:
                agg_dict["prob_shift"] = collapse_identical_variants
            df = (
                df[cols]
                .assign(weight=1)
                .groupby(["concentration", "aa_substitutions"], as_index=False)
                .aggregate(agg_dict)
            )

        # Add columns that parse mutations into wt amino acid, site,
        # and mutant amino acid
        ret_fs_df = func_score_df.reset_index()
        ret_fs_df["wts"], ret_fs_df["sites"], ret_fs_df["muts"] = zip(
            *ret_fs_df[substitution_col].map(split_subs)
        )
    
        # Use the substitution_col to infer the wildtype
        # amino-acid sequence of each condition, storing this
        # information in a dataframe.
        site_map = pandas.DataFrame(dtype="string")
        for hom, hom_func_df in ret_fs_df.groupby(condition_col):
            for idx, row in hom_func_df.iterrows():
                for wt, site  in zip(row.wts, row.sites):
                    site_map.loc[site, hom] = wt
        
        # Find all sites for which at least one condition lacks data
        # (this can happen if there is a gap in the alignment)
        na_rows = site_map.isna().any(axis=1)
        print(f"Found {sum(na_rows)} site(s) lacking data in at least one condition.")
        sites_to_throw = na_rows[na_rows].index
        site_map.dropna(inplace=True)
        
        # Remove all variants with a mutation at one of the above
        # "disallowed" sites lacking data
        def flags_disallowed(disallowed_sites, sites_list):
            """Check to see if a sites list contains 
            any disallowed sites"""
            for site in sites_list:
                if site in disallowed_sites:
                    return False
            return True
        
        ret_fs_df["allowed_variant"] = ret_fs_df.sites.apply(
            lambda sl: flags_disallowed(sites_to_throw,sl)
        )
        n_var_pre_filter = len(ret_fs_df)
        ret_fs_df = ret_fs_df[ret_fs_df["allowed_variant"]]
        print(f"{n_var_pre_filter-len(ret_fs_df)} of the {n_var_pre_filter} variants"
              f" were removed because they had mutations at the above sites, leaving"
              f" {len(ret_fs_df)} variants.")
    
        # Duplicate the substitutions_col, 
        # then convert the respective subs to be wrt ref
        # using the function above
        ret_fs_df = ret_fs_df.assign(var_wrt_ref = ret_fs_df[substitution_col])
        for hom, hom_func_df in ret_fs_df.groupby(condition_col):
            
            if hom == reference_condition: continue
            # TODO, conditions with identical site maps should share cache, yea?
            # this would greatly impove the analysis of replicate sequence conditions
            variant_cache = {} 
            cache_hits = 0
            
            for idx, row in tqdm(hom_func_df.iterrows(), total=len(hom_func_df)):
                
                key = tuple(list(zip(row.wts, row.sites, row.muts)))
                if key in variant_cache:
                    ret_fs_df.loc[idx, "var_wrt_ref"]  = variant_cache[key]
                    cache_hits += 1
                    continue
                
                var_map = site_map[[reference_condition, hom]].copy()
                for wt, site, mut in zip(row.wts, row.sites, row.muts):
                    var_map.loc[site, hom] = mut
                nis = var_map.where(
                    var_map[reference_condition] != var_map[hom]
                ).dropna()
                muts = nis[reference_condition] + nis.index + nis[hom]
                
                mutated_seq = " ".join(muts.values)
                ret_fs_df.loc[idx, "var_wrt_ref"] = mutated_seq
                variant_cache[key] = mutated_seq
                
            print(f"There were {cache_hits} cache hits in total for condition {hom}.")
    
        # Get list of all allowed substitutions for which we will tune beta parameters
        allowed_subs = {
            s for subs in ret_fs_df.var_wrt_ref
            for s in subs.split()
        }
        
        # Make BinaryMap representations for each condition
        X, y = {}, {}
        for condition, condition_func_score_df in ret_fs_df.groupby(condition_col):
            ref_bmap = bmap.BinaryMap(
                condition_func_score_df,
                substitutions_col="var_wrt_ref",
                allowed_subs=allowed_subs
            )
            
            # convert binarymaps into sparse arrays for model input
            X[condition] = sparse.BCOO.from_scipy_sparse(ref_bmap.binary_variants)
            
            # create jax array for functional score targets
            y[condition] = jnp.array(condition_func_score_df[func_score_col].values)
        
        ret_fs_df.drop(["wts", "sites", "muts"], axis=1, inplace=True)
    
        return (X, y), ret_fs_df, ref_bmap.all_subs, site_map 


    def _initialize_model_params(
            homologs: dict, 
            n_beta_shift_params: int,
            include_alpha=True,
            init_sig_range=10.,
            init_sig_min=-10.,
            latent_bias=5.
    ):
        """
        initialize a set of starting parameters for the JAX model.
        
        Parameters
        ----------
        
        homologs : list
            A list containing all possible target homolog 
            names.
        
        n_beta_shift_params: int
            The number of beta and shift parameters 
            (for each homolog) to initialize.
        
        include_alpha : book
            Initialize parameters for the sigmoid from
            the global epistasis function
        
        init_sig_range : float
            The range of observed phenotypes in the raw
            data, used to initialize the range of the
            sigmoid from the global epistasis function
        
        init_sig_min : float
            The lower bound of observed phenotypes in
            the raw data, used to initialize the minimum
            value of the sigmoid from the global epistasis
            funciton
            
        latent_bias : float
            bias parameter applied to the output of the
            linear model (latent prediction).
        
        Returns
        -------
        
        dict :
            all relevant parameters to be tuned with the JAX model.
        """
        
        params = {}
        seed = 0
        key = jax.random.PRNGKey(seed)

        # initialize beta parameters from normal distribution.
        params["β"] = jax.random.normal(shape=(n_beta_shift_params,), key=key)

        # initialize shift parameters
        for homolog in homologs:
            # We expect most shift parameters to be close to zero
            params[f"S_{homolog}"] = jnp.zeros(shape=(n_beta_shift_params,))
            params[f"C_{homolog}"] = jnp.zeros(shape=(1,))
            params[f"γ_{homolog}"] = jnp.zeros(shape=(1,))

        if include_alpha:
            params["α"]=dict(
                ge_scale=jnp.array([init_sig_range]),
                ge_bias=jnp.array([init_sig_min])
            )
            
        params["C_ref"] = jnp.array([5.0]) # 5.0 is a guess, could update

        return params


    # TODO This could be a nice spot to get some useful information about
    # wildtypes of each condition
    @property
    def wt_df(self):
        r"""pandas.DataFrame: Activities :math:`a_{\rm{wt,e}}` for conditions."""
        pass

    @property
    def mutation_df(self):
        r"""pandas.DataFrame: Escape :math:`\beta_{m,e}` for each mutation."""
        pass

        # polyclonal

    @property
    def mutation_site_summary_df(
        self,
        *,
        min_times_seen=1,
        mutation_whitelist=None,
        exclude_chars=frozenset(["*"]),
    ):
        """Site-level summaries of mutation shift.

        Parameters
        ----------
        min_times_seen : int
            Only include in summaries mutations seen in at least this many variants.
        mutation_whitelist : None or set
            Only include in summaries these mutations.
        exclude_chars : set or list
            Exclude mutations to these characters when calculating site summaries.
            Useful if you want to ignore stop codons (``*``), and perhaps in some
            cases also gaps (``-``).

        Returns
        -------
        pandas.DataFrame

        """
        pass


    # TODO Set up f(v, h) = t(g(ϕ(v, h)))
    def predict():
        pass

    # TODO
    def fit(self):
        pass 


    # TODO all plotting
    def activity_wt_barplot(self, **kwargs):
        r"""Bar plot of activity against each condition, :math:`a_{\rm{wt},e}`.

        Parameters
        ----------
        **kwargs
            Keyword args for :func:`multidms.plot.activity_wt_barplot`.

        Returns
        -------
        altair.Chart
            Interactive plot.

        """
        pass

        """
        kwargs["activity_wt_df"] = self.activity_wt_df
        if "condition_colors" not in kwargs:
            kwargs["condition_colors"] = self.condition_colors
        return multidms.plot.activity_wt_barplot(**kwargs)
        """

    # TODO
    def filter_variants_by_seen_muts(
        self,
        variants_df,
        min_times_seen=1,
        subs_col="aa_substitutions",
    ):
        """Remove variants that contain mutations not seen during model fitting.

        Parameters
        ----------
        variants_df : pandas.DataFrame
            Contains variants as rows.
        min_times_seen : int
            Require mutations to be seen >= this many times in data used to fit model.
        subs_col : str
            Column in `variants_df` with mutations in each variant.

        Returns
        -------
        variants_df : pandas.DataFrame
            Copy of input dataframe, with rows of variants
            that have unseen mutations removed.
        """
        pass

        """
        variants_df = variants_df.copy()

        if subs_col not in variants_df.columns:
            raise ValueError(f"`variants_df` lacks column {subs_col}")

        filter_col = "_pass_filter"
        if filter_col in variants_df.columns:
            raise ValueError(f"`variants_df` cannot have column {filter_col}")

        if min_times_seen == 1:
            allowed_muts = self.mutations
        elif self.mutations_times_seen is not None:
            allowed_muts = {
                m for (m, n) in self.mutations_times_seen.items() if n >= min_times_seen
            }
        else:
            raise ValueError(f"Cannot use {min_times_seen=} without data to fit")

        variants_df[filter_col] = variants_df[subs_col].map(
            lambda s: set(s.split()).issubset(allowed_muts)
        )

        return (
            variants_df.query("_pass_filter == True")
            .drop(columns="_pass_filter")
            .reset_index(drop=True)
        )
        """

    # TODO
    def _get_binarymap(
        self,
        variants_df,
    ):
        """Get ``BinaryMap`` appropriate for use."""
        pass

        """
        bmap = binarymap.BinaryMap(
            variants_df,
            substitutions_col="aa_substitutions",
            allowed_subs=self.mutations,
            alphabet=self.alphabet,
            sites_as_str=not self.sequential_integer_sites,
        )
        if tuple(bmap.all_subs) != self.mutations:
            raise ValueError(
                "Different mutations in BinaryMap and self:"
                f"\n{bmap.all_subs=}\n{self.mutations=}"
            )
        return bmap
        """

    # TODO
    def mut_shift_corr(self, ref_poly):
        """Correlation of mutation-shift values with another model.

        For each condition, how well is this model's mutation-shift values
        correlation with another model?

        Mutations present in only one model are ignored.

        Parameters
        ------------
        ref_poly : :class:`Multidms`
            Other (reference) multidms model with which we calculate correlations.

        Returns
        ---------
        corr_df : pandas.DataFrame
            Pairwise condition correlations for shift.
        """
        pass

        """
        if self.mut_shift_df is None or ref_poly.mut_escape_df is None:
            raise ValueError("Both objects must have `mut_shift_df` initialized.")

        df = pandas.concat(
            [
                self.mut_shift_df.assign(
                    condition=lambda x: list(zip(itertools.repeat("self"), x["condition"]))
                ),
                ref_poly.mut_shift_df.assign(
                    condition=lambda x: list(zip(itertools.repeat("ref"), x["condition"]))
                ),
            ]
        )

        corr = (
            multidms.utils.tidy_to_corr(
                df,
                sample_col="condition",
                label_col="mutation",
                value_col="shift",
            )
            .assign(
                ref_condition=lambda x: x["condition_2"].map(lambda tup: tup[1]),
                ref_model=lambda x: x["condition_2"].map(lambda tup: tup[0]),
                self_condition=lambda x: x["condition_1"].map(lambda tup: tup[1]),
                self_model=lambda x: x["condition_1"].map(lambda tup: tup[0]),
            )
            .query("ref_model != self_model")[
                ["ref_condition", "self_condition", "correlation"]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        return corr
        """


if __name__ == "__main__":
    pass
    # TODO
    #import doctest
    #doctest.testmod()
