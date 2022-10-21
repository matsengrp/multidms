"""
==========
multidms
==========

Defines :class:`Multidms` objects for handling data from one or more
dms experiments under various conditions.

"""

from functools import partial as o_partial
# from jax.tree_util import Partial
import collections
import copy  # noqa: F401
import inspect
import itertools
import os
import sys
import time
import json

# bloom lab tools
import binarymap as bmap
from polyclonal.plot import DEFAULT_POSITIVE_COLORS
# TODO https://github.com/google/jax/issues/3045 JIT MutationParser functions?
from polyclonal.utils import MutationParser
import frozendict
# import natsort
import numpy as onp
import pandas
from tqdm import tqdm

# jax
# TODO do we need the cuda version, as well?
import jax
import jaxlib
# from jax import jit
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental import sparse
import jaxopt
from jaxopt import ProximalGradient

# local
# TODO import only what you need, here.
import multidms
#import multidms.plot
import multidms.utils
from multidms.model import identity_activation


class stub_MultidmsFitError(Exception):
    """Error fitting in :meth:`Multidms.fit`."""

    pass

# TODO update model description based upon hugh's recent updates
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
    
    In the matrix algebra, the sum of $β\ *m$ and $S*\ {m,h}$ 
    gives a vector of mutational effects, with one entry per mutation.
    Multiplying the matrix $X_h$ by this vector gives a new 
    vector with one entry per variant, where values are the 
    sum of mutational effects, weighted by the variant-specific weights in $X_h$.
    Adding the $c$ value to this vector will give a vector of 
    predicted latent phenotypes for each variant.
    
    Next, the global-epistasis function can be used to convert 
        a vector of predicted latent phenotypes to a vector of 
        predicted functional scores.
    
    $$F\ *{h,pred} = g*\ {\alpha}(P_h)$$
    
    Finally, this vector could be fed into a loss function and 
    compared with a vector of observed functional scores.

    Note
    ----
    You can initialize a :class:`Multidms` object with variants data from one or
    more conditions, as well as a jit-compiled predictive function, objective
    function to minimize, and a proximal function for non smooth constraints on
    the model.

    1. functional score dataframe
        TODO

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
    mutations_df : pandas.DataFrame
        1. mutation_shifts : TODO
        2. mutation_betas : TODO
        3. mutation_functional_effects : TODO
    reference : str
        The reference factor level in conditions. All mutations will be converted to
        be with respect to this contition's inferred wildtype sequence. See
        The class description for more.
    alphabet : tuple 
        Allowed characters in mutation strings.
    site_map : tuple 
        Inferred from ``data_to_fit``, this attribute will provide the wildtype
        amino acid at all sites, for all conditions.
    letter_suffixed_sites=False,
        True if sites are allowed to have trailing charicter suffixes, False otherwise.
    condition_colors : dict TODO
        Maps each condition to its color.
    data_to_fit : pandas.DataFrame
        Data to fit as passed when initializing this :class:`Multidms` object.
        If using ``collapse_identical_variants``, then identical variants
        are collapsed on columns 'concentration', 'aa_substitutions',
        and 'prob_shift', and a column 'weight' is added to represent number
        of collapsed variants. Also, row-order may be changed.

    Example
    -------
    Simple example with two conditions (`1` and `2`)

    >>> import pandas as pd
    >>> import multidms
    >>> func_score_data = {
    ...     'condition' : ["1","1","1","1", "2","2","2","2","2","2"],
    ...     'aa_substitutions' : ['M1E', 'G3R', 'G3P', 'M1W', 'M1E', 'P3R', 'P3G', 'M1E P3G', 'M1E P3R', 'P2T'],
    ...     'func_score' : [2, -7, -0.5, 2.3, 1, -5, 0.4, 2.7, -2.7, 0.3],
    ... }
    >>> func_score_df = pd.DataFrame(func_score_data)
    >>> func_score_df
      condition aa_substitutions  func_score
      0         1              M1E         2.0
      1         1              G3R        -7.0
      2         1              G3P        -0.5
      3         1              M1W         2.3
      4         2              M1E         1.0
      5         2              P3R        -5.0
      6         2              P3G         0.4
      7         2          M1E P3G         2.7
      8         2          M1E P3R        -2.7
      9         2              P2T         0.3

    >>> from multidms.model import global_epistasis

    Using the predcompiled model, `global epistasis`, we can initialize the 
    `Multidms` Object

    >>> mdms = multidms.Multidms(
    ...     func_score_df,
    ...     *multidms.model.global_epistasis.values(),
    ...     alphabet = multidms.AAS_WITHSTOP,
    ...     reference = "1"
    ... )

    This object initializes a few useful attributes and properties

    >>> mdms.conditions
    ('1', '2')

    >>> mdms.mutations
    ('M1E', 'M1W', 'G3P', 'G3R')

    >>> mdms.site_map
       1  2
       3  G  P
       1  M  M

    >>> mdms.mutations_df
      mutation         β wts  sites muts  times_seen  S_1       F_1  S_2       F_2
      0      M1E  0.080868   M      1    E           4  0.0 -0.061761  0.0 -0.061761
      1      M1W -0.386247   M      1    W           1  0.0 -0.098172  0.0 -0.098172
      2      G3P -0.375656   G      3    P           2  0.0 -0.097148  0.0 -0.097148
      3      G3R  1.668974   G      3    R           3  0.0 -0.012681  0.0 -0.012681


    >>> mdms.data_to_fit
      condition aa_substitutions  ...  predicted_func_score  corrected_func_score
      0         1              G3P  ...             -0.097148                  -0.5
      1         1              G3R  ...             -0.012681                  -7.0
      2         1              M1E  ...             -0.061761                   2.0
      3         1              M1W  ...             -0.098172                   2.3
      4         2              M1E  ...             -0.089669                   1.0
      5         2          M1E P3G  ...             -0.061761                   2.7
      6         2          M1E P3R  ...             -0.011697                  -2.7
      8         2              P3G  ...             -0.066929                   0.4
      9         2              P3R  ...             -0.012681                  -5.0


    We can then fit the data.

    >>> data = (mdms.binarymaps['X'], mdms.binarymaps['y'])
    >>> compiled_cost = global_epistasis["objective"]
    >>> compiled_cost(mdms.params, data)
    4.434311992312495
    >>> mdms.fit()
    >>> compiled_cost(mdms.params, data)
    0.3332387869442089
    """

    # TODO sites : array-like or 'infer' #2
    # Offer ability to provide your own refernce sites?
    # Right now the inference seems like the best way to go?
    # except, it seems to me that we always want these models to
    # to be sparse unless we want to predict?

    # TODO *, ? What's with this parameter? I'm assuming it's for
    # child classes.

    # TODO condition_colors : array-like or dict
    # Offer option to provide condition colors, or cmap / palette?

    # TODO, what if we made classes for objective functions 
    # etc, that gave us the compiled functions, as well as 
    # attributes that define then i.e. number of parameters
    # inputs and outputs needed etc?

    def __init__(
        self,
        data_to_fit : pandas.DataFrame,
        predict_function,
        objective_function,
        proximal_function,
        reference : str,
        alphabet=multidms.AAS,
        init_missing="zero",
        letter_suffixed_sites=False,
        condition_colors=DEFAULT_POSITIVE_COLORS,
        **kwargs
    ):
        """See main class docstring."""
        self._predict_function = predict_function
        self._objective_function = objective_function
        self._proximal_function = proximal_function

        # check init seed 
        # TODO Is this necessary?
        if isinstance(init_missing, int):
            # TODO set the JAX PRNG key?
            numpy.random.seed(init_missing)
        elif init_missing != "zero":
            raise ValueError(f"invalid {init_missing=}")    

        # Check and initialize conditions attribute
        if pandas.isnull(data_to_fit["condition"]).any():
            raise ValueError("condition name cannot be null")
        self.conditions = tuple(data_to_fit["condition"].unique())

        if reference not in self.conditions:
            raise ValueError("reference must be in condition factor levels")
        self.reference = reference

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
            letter_suffixed_sites
        )


        # Check and initialize fitting data from func_score_df
        (
            self.binarymaps,
            self._data_to_fit,
            self.mutations,
            self.site_map,
        ) = self._create_condition_modeling_data(
            data_to_fit, 
            reference,
            **kwargs    
        )

        # set internal params with activities and shifts
        # TODO This should rely on the three functions passed in
        self.params = self._initialize_model_params(
            self._data_to_fit["condition"].unique(), 
            n_beta_shift_params=self.binarymaps['X'][reference].shape[1],
            include_alpha=True, # TODO would could just initialize them after ... ?
            #init_sig_range=sig_range, #TODO
            #init_sig_min=sig_lower
        )

        # TODO make this a property
        # initialize single mutational effects df
        mut_df = pandas.DataFrame(
            {
                "mutation" : self.mutations,
                "β" : self.params["β"]
            }
        )

        # TODO JIT
        #parser = partial(multidms.utils.split_sub, parser=self._mutparser.parse_mut)
        mut_df["wts"], mut_df["sites"], mut_df["muts"] = zip(
            *mut_df["mutation"].map(self._mutparser.parse_mut)
        )

        # compute times seen in data
        # TODO Should this be seen times per condition?
        #times_seen_per_condition = {}
        #for condition, condition_df in self._data_to_fit.groupby("condition"):
        times_seen = (
            self._data_to_fit["var_wrt_ref"]
            .str.split()
            .explode()
            .value_counts()
        )
        if (times_seen == times_seen.astype(int)).all():
            times_seen = times_seen.astype(int)
        times_seen.index.name = f"mutation"
        times_seen.name = f"times_seen"
        mut_df = mut_df.merge(
                times_seen, 
                left_on="mutation", right_on="mutation", 
                how="outer" 
        )

        self._mutations_df = mut_df

        # add other current model properties.
        # self._update_model_effects_dfs()

    @property
    def mutations_df(self):
        """ Get all mutational attributes with the current parameters """

        # update the betas
        self._mutations_df.loc[:, "β"] = self.params["β"]

        # make predictions
        binary_single_subs = sparse.BCOO.fromdense(onp.identity(len(self.mutations)))
        for condition in self.conditions:
            
            # collect relevant params
            h_params = self.get_condition_params(condition)

            # attach relevent params to mut effects df
            self._mutations_df[f"S_{condition}"] = self.params[f"S_{condition}"]
            
            # predictions for all single subs
            # TODO how should we handle fitting/prediction hyper params i.e. kwargs? 
            self._mutations_df[f"F_{condition}"] = self._predict_function(
                h_params, 
                binary_single_subs
                # self._ϕ, self._g, self._t,
                # **kwargs
            )

        return self._mutations_df

    @property
    def data_to_fit(self):
        """ Get all mutational attributes with the current parameters """

        self._data_to_fit["predicted_latent_phenotype"] = onp.nan
        self._data_to_fit[f"predicted_func_score"] = onp.nan
        self._data_to_fit[f"corrected_func_score"] = self._data_to_fit[f"func_score"]
        for condition, condition_dtf in self._data_to_fit.groupby("condition"):

            # TODO how should we handle fitting/prediction hyper params i.e. kwargs? 
            h_params = self.get_condition_params(condition)
            y_h_pred = self._predict_function(
                h_params, 
                self.binarymaps['X'][condition]
                # **kwargs
            )

            self._data_to_fit.loc[condition_dtf.index, f"predicted_func_score"] = y_h_pred
            self._data_to_fit.loc[condition_dtf.index, f"corrected_func_score"] -= h_params[f"γ"]
            
        return self._data_to_fit
        
    # TODO impliment different condition overlap options
    # TODO move the column names to a config of some kind. JSON?
    # TODO Break this up into jit-able helper functions and move to init
    def _create_condition_modeling_data(
        self,
        func_score_df:pandas.DataFrame,
        reference_condition:str,
        # condition_overlap="sites", # TODO document
        collapse_identical_variants="mean",
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

        cols = [
            condition_col,
            substitution_col,
            func_score_col
        ]
        if "weight" in func_score_df.columns:
            cols.append(
                "weight"
            )  # will be overwritten if `collapse_identical_variants`
        if not func_score_df[cols].notnull().all().all():
            raise ValueError(f"null entries in data frame of variants:\n{df[cols]}")

        if collapse_identical_variants:
            agg_dict = {
                "weight" : "sum", 
                func_score_col : collapse_identical_variants
            }
            #if get_pv:
            #    agg_dict["prob_shift"] = collapse_identical_variants
            df = (
                func_score_df[cols]
                .assign(weight=1)
                .groupby([condition_col, "aa_substitutions"], as_index=False)
                .aggregate(agg_dict)
            )

        # Add columns that parse mutations into wt amino acid, site,
        # and mutant amino acid
        else:
            df = func_score_df.reset_index()

        # TODO JIT
        parser = o_partial(multidms.utils.split_subs, parser=self._mutparser.parse_mut)
        df["wts"], df["sites"], df["muts"] = zip(
            *df[substitution_col].map(parser)
        )
    
        # Use the substitution_col to infer the wildtype
        # amino-acid sequence of each condition, storing this
        # information in a dataframe.
        site_map = pandas.DataFrame()
        for hom, hom_func_df in df.groupby(condition_col):
            for idx, row in hom_func_df.iterrows():
                for wt, site  in zip(row.wts, row.sites):
                    site_map.loc[site, hom] = wt
        
        # Find all sites for which at least one condition lacks data
        # (this can happen if there is a gap in the alignment)
        na_rows = site_map.isna().any(axis=1)
        # TODO let's move any print statements to 
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
        
        df["allowed_variant"] = df.sites.apply(
            lambda sl: flags_disallowed(sites_to_throw,sl)
        )
        n_var_pre_filter = len(df)
        df = df[df["allowed_variant"]]
        print(f"{n_var_pre_filter-len(df)} of the {n_var_pre_filter} variants"
              f" were removed because they had mutations at the above sites, leaving"
              f" {len(df)} variants.")
    
        # Duplicate the substitutions_col, 
        # then convert the respective subs to be wrt ref
        # using the function above
        df = df.assign(var_wrt_ref = df[substitution_col])
        for hom, hom_func_df in df.groupby(condition_col):
            
            if hom == reference_condition: continue
            # TODO, conditions with identical site maps should share cache, yea?
            # this would greatly impove the analysis of replicate sequence conditions
            variant_cache = {} 
            cache_hits = 0
            
            for idx, row in tqdm(hom_func_df.iterrows(), total=len(hom_func_df)):
                
                key = tuple(list(zip(row.wts, row.sites, row.muts)))
                if key in variant_cache:
                    df.loc[idx, "var_wrt_ref"]  = variant_cache[key]
                    cache_hits += 1
                    continue
                
                var_map = site_map[[reference_condition, hom]].copy()
                for wt, site, mut in zip(row.wts, row.sites, row.muts):
                    var_map.loc[site, hom] = mut
                nis = var_map.where(
                    var_map[reference_condition] != var_map[hom]
                ).dropna().astype(str)
                
                muts = nis[reference_condition] + nis.index.astype(str) + nis[hom]
                
                mutated_seq = " ".join(muts.values)
                df.loc[idx, "var_wrt_ref"] = mutated_seq
                variant_cache[key] = mutated_seq
                
            print(f"There were {cache_hits} cache hits in total for condition {hom}.")
    
        # Get list of all allowed substitutions for which we will tune beta parameters
        allowed_subs = {
            s for subs in df.var_wrt_ref
            for s in subs.split()
        }
        
        # Make BinaryMap representations for each condition
        X, y = {}, {}
        for condition, condition_func_score_df in df.groupby(condition_col):
            ref_bmap = bmap.BinaryMap(
                condition_func_score_df,
                substitutions_col="var_wrt_ref",
                allowed_subs=allowed_subs
            )
            
            # convert binarymaps into sparse arrays for model input
            X[condition] = sparse.BCOO.from_scipy_sparse(ref_bmap.binary_variants)
            
            # create jax array for functional score targets
            y[condition] = jnp.array(condition_func_score_df[func_score_col].values)
        
        df.drop(["wts", "sites", "muts"], axis=1, inplace=True)

        # TODO should we separate binarymaps and targets?
        return {'X':X, 'y':y}, df, tuple(ref_bmap.all_subs), site_map 


    def _initialize_model_params(
            self,
            conditions: dict, 
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
        
        conditions : list
            A list containing all possible target condition 
            names.
        
        n_beta_shift_params: int
            The number of beta and shift parameters 
            (for each condition) to initialize.
        
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
        for condition in conditions:
            # We expect most shift parameters to be close to zero
            params[f"S_{condition}"] = jnp.zeros(shape=(n_beta_shift_params,))
            params[f"C_{condition}"] = jnp.zeros(shape=(1,))
            params[f"γ_{condition}"] = jnp.zeros(shape=(1,))

        # TODO Do we need this?
        if include_alpha:
            params["α"]=dict(
                ge_scale=jnp.array([init_sig_range]),
                ge_bias=jnp.array([init_sig_min])
            )
            
        # TODO
        params["C_ref"] = jnp.array([5.0]) # 5.0 is a guess, could update

        return params


    def get_condition_params(self, condition=None):
        """ get the relent parameters for a model prediction"""

        condition = self.reference if condition is None else condition
        if condition not in self.conditions:
            raise ValueError("condition does not exist in model")

        return {
            "α":self.params[f"α"],
            "β":self.params["β"], 
            "C_ref":self.params["C_ref"],
            "S":self.params[f"S_{condition}"], 
            "C":self.params[f"C_{condition}"],
            "γ":self.params[f"γ_{condition}"]
        }


    # TODO finish documentation.
    def condition_predict(self, X, condition=None):
        """ condition specific prediction on X using the biophysical model
        given current model parameters. """

        # TODO assert X is correct shape.
        # TODO assert that the substitutions exist?
        # TODO require the user
        h_params = get_condition_params(condition)
        return self._predict_function(h_params, X)
    

    # TODO finish documentation.
    # TODO lasso etc paramerters (**kwargs ?)
    def fit(
        self, 
        λ_lasso=1e-5,
        λ_ridge=0,
        **kwargs
    ):
        """ use jaxopt.ProximalGradiant to optimize parameters on
        `self._data_to_fit` 
        """
        # Use partial 
        # compiled_smooth_cost = Partial(smooth_cost, self._predict_function)

        solver = ProximalGradient(
            self._objective_function,
            self._proximal_function,
            tol=1e-6,
            maxiter=1000
        )

        # the reference shift and gamma parameters forced to be zero
        lock_params = {
            f"S_{self.reference}" : jnp.zeros(len(self.params['β'])),
            f"γ_{self.reference}" : jnp.zeros(shape=(1,)),
        }

        # currently we lock C_h because of odd model behavior
        for condition in self.conditions:
            lock_params[f"C_{condition}"] = jnp.zeros(shape=(1,))

        # lasso regularization on the Shift parameters
        lasso_params = {}
        for non_ref_condition in self.conditions:
            if non_ref_condition == self.reference: continue
            lasso_params[f"S_{non_ref_condition}"] = λ_lasso

        # run the optimizer
        self.params, state = solver.run(
            self.params,
            hyperparams_prox = dict(
                lasso_params = lasso_params,
                lock_params = lock_params
            ),
            data=(self.binarymaps['X'], self.binarymaps['y']),
            λ_ridge=λ_ridge,
            **kwargs
            #lower_bound=fit_params['lower_bound'],
            #hinge_scale=fit_params['hinge_scale']
        )
        

    # TODO
    # wildtypes of each condition
    @property
    def stub_wt_df(self):
        r"""pandas.DataFrame: Activities :math:`a_{\rm{wt,e}}` for conditions.
        This could be a nice spot to get some useful information about wts 
        of each condition"""
        pass

    @property
    def stub_mutation_site_summary_df(
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



    # TODO all plotting
    def plot(self, **kwargs):
        pass

    # TODO
    def filter_variants_by_seen_muts(self):
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


    # TODO
    def _get_binarymap(
        self,
        variants_df,
    ):
        pass

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


if __name__ == "__main__":
    pass
    # TODO
    #import doctest
    #doctest.testmod()
