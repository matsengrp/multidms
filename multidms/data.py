"""
==========
multidms
==========

Defines :class:`Multidms` objects for handling data from one or more
dms experiments under various conditions.

"""


# external
import binarymap as bmap
from polyclonal.plot import DEFAULT_POSITIVE_COLORS
from polyclonal.utils import MutationParser
import numpy as onp
import pandas
from tqdm import tqdm
import jax
import jaxlib
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental import sparse
import jaxopt
from jaxopt import ProximalGradient
from frozendict import frozendict
from collections import namedtuple

# local
from functools import partial
import multidms.utils


# TODO Logging and Error Handling?
# TODO plotting methods for data summary.
class MultiDmsData:
    r"""
    Prep data for multidms model(s),
    Summarize, and provide 
    common attributes.

    Note
    ----
    You can initialize a :class:`Multidms` object with a ``pd.DataFrame``
    with a row for each variant sampled and annotations
    provided in the required columns:

    1. `condition` - Experimental condition from
        which a sample measurement was obtained.
    2. `aa_substitutions` - Defines each variant
        :math:`v` as a string of substitutions (e.g., 'M3A K5G').
        Note that while conditions may have differing wild types
        at a given site, the sites between conditions should reference
        the same site when alignment is performed between 
        condition wild types.
    3. `func_score` - The functional score computed from experimental
        measurements.

    Parameters
    ----------
    variants_df : pandas.DataFrame or None
        The variant level information from all experiments you
        wish to analyze. Should have columns named 'condition', 
        'aa_substitutions', and 'func_score'.
        See the class note for descriptions of each of the features.
    reference : str
        Name of the factor level which annotates the reference condition
        variants. Note that for model fitting this class will convert all
        amino acid substitutions for non-reference condition groups
        to relative to the reference condition. 
        For example, if the wild type amino acid at site 30 is an 
        A in the reference condition, and a G in a non-reference condition, 
        then a Y30G mutation in the non-reference condition is recorded as an A30G 
        mutation relative to the reference. This way, each condition informs 
        the exact same parameters, even at sites that differ in wild type amino acid.
        These are encoded in a ``BinaryMap`` object for each condtion, 
        where all sites that are non-identical to the reference are 1's.
        For motivation, see the `Model overview` section in `multidms.MultiDmsModel`
        class notes.
    collapse_identical_variants : {'mean', 'median', False}
        If identical variants in ``variants_df`` (same 'aa_substitutions'),
        exist within individual condition groups,
        collapse them by taking mean or median of 'func_score', or
        (if `False`) do not collapse at all. Collapsing will make fitting faster,
        but *not* a good idea if you are doing bootstrapping.
    alphabet : array-like
        Allowed characters in mutation strings.
    condition_colors : array-like or dict
        Maps each condition to the color used for plotting. Either a dict keyed
        by each condition, or an array of colors that are sequentially assigned
        to the conditions.
    letter_suffixed_sites: bool
        True if sites are sequential and integer, False otherwise.


    Attributes
    ----------
    variants_df : pandas.DataFrame
        Data to fit as passed when initializing this :class:`Multidms` object
        less those variants which were thrown due to mutations
        outside the union of sites seen in across all condition variants.
        If using ``collapse_identical_variants``, then identical variants
        are collapsed on columns 'condition', 'aa_substitutions', 
        and a column 'weight' is added to represent number
        of collapsed variants. 
        Also, row-order may be changed.
    reference : str
        The reference factor level in conditions. All mutations will be converted to
        be with respect to this condition's inferred wild type sequence. See
        The class description for more.
    conditions : tuple
        Names of all conditions.
    condition_colors : dict TODO
        Maps each condition to its color.
    alphabet : tuple 
        Allowed characters in mutation strings.
    site_map : tuple 
        Inferred from ``variants_df``, this attribute will provide the wild type
        amino acid at all sites, for all conditions.
    mutations : pandas.DataFrame
        A dataframe containing all relevant substitutions and all relevant.
    mutations_df : pandas.DataFrame
        1. mutation_shifts : TODO
        2. mutation_betas : TODO
        3. mutation_functional_effects : TODO

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

    Instantiate a ``MultiDmsData`` Object allowing for stop codon variants
    and declaring condition '1' as the reference condition.

    >>> data = multidms.Multidms(
    ...     func_score_df,
    ...     alphabet = multidms.AAS_WITHSTOP,
    ...     reference = "1"
    ... )

    Note this may take some time due to the string
    operations that must be performed when converting
    amino acid substitutions to be with respect to a
    reference wild type sequence.
    After the object has finished being instantiated,
    we now have access to a few 'static' attributes
    of our data.

    >>> data.reference
    '1'

    >>> data.conditions
    ('1', '2')

    >>> data.mutations
    ('M1E', 'M1W', 'G3P', 'G3R')

    >>> data.site_map
       1  2
       3  G  P
       1  M  M

    >>> data.mutations_df
      mutation         β wts  sites muts  times_seen  S_1       F_1  S_2       F_2
      0      M1E  0.080868   M      1    E           4  0.0 -0.061761  0.0 -0.061761
      1      M1W -0.386247   M      1    W           1  0.0 -0.098172  0.0 -0.098172
      2      G3P -0.375656   G      3    P           2  0.0 -0.097148  0.0 -0.097148
      3      G3R  1.668974   G      3    R           3  0.0 -0.012681  0.0 -0.012681


    >>> data.variants_df
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
    """

    # TODO sites : array-like or 'infer' #2
    # Offer ability to provide your own refernce sites?
    # Maybe we should also
    # Right now the inference seems like the best way to go?
    # except, it seems to me that we always want these models to
    # to be sparse unless we want to predict?

    # TODO what about the behavior to expect for throwing variants
    # i.e. Currently, we throw all variants which include substitutions
    # on sites that are not seen across all conditions. We could
    # also provide a 


    def __init__(
        self,
        variants_df: pandas.DataFrame,
        reference : str,
        alphabet=multidms.AAS,
        collapse_identical_variants="mean",
        condition_colors=DEFAULT_POSITIVE_COLORS,
        letter_suffixed_sites=False
    ):
        """See main class docstring."""

        # Check and initialize conditions attribute
        if pandas.isnull(variants_df["condition"]).any():
            raise ValueError("condition name cannot be null")
        self._conditions = tuple(variants_df["condition"].unique())

        if reference not in self._conditions:
            raise ValueError("reference must be in condition factor levels")
        self._reference = reference

        self._collapse_identical_variants = collapse_identical_variants

        # Check and initialize condition colors
        if isinstance(condition_colors, dict):
            self.condition_colors = {e: condition_colors[e] for e in self._conditions}
        elif len(condition_colors) < len(self._conditions):
            raise ValueError("not enough `condition_colors`")
        else:
            self.condition_colors = dict(zip(self._conditions, condition_colors))
        
        # Check and initialize alphabet & mut parser attributes
        if len(set(alphabet)) != len(alphabet):
            raise ValueError("duplicate letters in `alphabet`")
        self.alphabet = tuple(alphabet)

        self._mutparser = MutationParser(
            alphabet,
            letter_suffixed_sites
        )

        # TODO Break this up into (jit-able) helper functions and move to init
        (
            self._training_data,
            self._binarymaps,
            self._variants_df,
            self._mutations,
            self._site_map,
        ) = self._create_condition_modeling_data(
            variants_df
        )

        # initialize single mutational effects df
        mut_df = pandas.DataFrame({"mutation" : self._mutations})

        mut_df["wts"], mut_df["sites"], mut_df["muts"] = zip(
            *mut_df["mutation"].map(self._mutparser.parse_mut)
        )

        # compute times seen in data
        for condition, condition_vars in self._variants_df.groupby("condition"):
            times_seen = (
                condition_vars["var_wrt_ref"]
                .str.split()
                .explode()
                .value_counts()
            )
            if (times_seen == times_seen.astype(int)).all():
                times_seen = times_seen.astype(int)
            times_seen.index.name = f"mutation"
            times_seen.name = f"times_seen_{condition}"
            mut_df = mut_df.merge(
                    times_seen, 
                    left_on="mutation", right_on="mutation", 
                    how="outer" 
            ).fillna(0)

        self._mutations_df = mut_df
        non_identical_mutations = {}
        non_identical_sites = {}
        for condition in self._conditions:

            if condition == self._reference:
                non_identical_mutations[condition] = ""
                non_identical_sites[condition] = []
                continue

            nis = self._site_map.where(
                self._site_map[self._reference] != self._site_map[condition]
            ).dropna().astype(str)
            muts = nis[self._reference] + nis.index.astype(str) + nis[condition]
            muts_string = " ".join(muts.values)
            non_identical_mutations[condition] = muts_string
            non_identical_sites[condition] = nis.index.values

        self._non_identical_mutations = frozendict(non_identical_mutations)
        self._non_identical_sites = frozendict(non_identical_sites)


    @property
    def non_identical_mutations(self):
        return self._non_identical_mutations


    @property
    def non_identical_sites(self):
        return self._non_identical_sites


    @property
    def conditions(self):
        return self._conditions


    @property
    def reference(self):
        return self._reference


    @property
    def mutations(self): 
        return self._mutations


    @property
    def mutations_df(self):
        return self._mutations_df


    @property
    def variants_df(self):
        return self._variants_df


    @property
    def site_map(self):
        return self._site_map


    @property
    def training_data(self):
        return self._training_data


    @property
    def binarymaps(self):
        return self._binarymaps


    @property
    def targets(self):
        return self._training_data['y']


    @property
    def mut_parser(self):
        return self._mut_parser

    # TODO move the column names to a config of some kind. JSON?
    def _create_condition_modeling_data(
        self,
        func_score_df:pandas.DataFrame
    ):
        """
        data prep helper.

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

        # Configure new variants df
        cols = [
            "condition",
            "aa_substitutions",
            "func_score"
        ]
        if "weight" in func_score_df.columns:
            cols.append(
                "weight"
            )  # will be overwritten if `self._collapse_identical_variants`
        if not func_score_df[cols].notnull().all().all():
            raise ValueError(f"null entries in data frame of variants:\n{df[cols]}")

        if self._collapse_identical_variants:
            agg_dict = {
                "weight" : "sum", 
                "func_score" : self._collapse_identical_variants
            }
            df = (
                func_score_df[cols]
                .assign(weight=1)
                .groupby(["condition", "aa_substitutions"], as_index=False)
                .aggregate(agg_dict)
            )

        else:
            df = func_score_df.reset_index()

        parser = partial(multidms.utils.split_subs, parser=self._mutparser.parse_mut)
        df["wts"], df["sites"], df["muts"] = zip(
            *df["aa_substitutions"].map(parser)
        )
    
        # Use the "aa_substitutions" to infer the 
        # wild type for each condition
        site_map = pandas.DataFrame()
        for hom, hom_func_df in df.groupby("condition"):
            for idx, row in hom_func_df.iterrows():
                for wt, site  in zip(row.wts, row.sites):
                    site_map.loc[site, hom] = wt
        
        # Throw variants if the contain non overlapping
        # mutations with all other conditions ...
        na_rows = site_map.isna().any(axis=1)
        sites_to_throw = na_rows[na_rows].index
        site_map.dropna(inplace=True)
        
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
    
        # convert the respective subs to be wrt reference
        df = df.assign(var_wrt_ref = df["aa_substitutions"])
        for condition, condition_func_df in df.groupby("condition"):
            
            if condition == self._reference: continue
            # TODO Conditions with identical wild types should share a cache.
            variant_cache = {} 
            cache_hits = 0
            
            for idx, row in tqdm(condition_func_df.iterrows(), total=len(condition_func_df)):
                
                key = tuple(list(zip(row.wts, row.sites, row.muts)))
                if key in variant_cache:
                    df.loc[idx, "var_wrt_ref"]  = variant_cache[key]
                    cache_hits += 1
                    continue
                
                var_map = site_map[[self._reference, condition]].copy()
                for wt, site, mut in zip(row.wts, row.sites, row.muts):
                    var_map.loc[site, condition] = mut
                #nis = self._non_identical_sites[condition]
                nis = var_map.where(
                    var_map[self._reference] != var_map[condition]
                ).dropna().astype(str)
                
                muts = nis[self._reference] + nis.index.astype(str) + nis[condition]
                
                mutated_seq = " ".join(muts.values)
                df.loc[idx, "var_wrt_ref"] = mutated_seq
                variant_cache[key] = mutated_seq
  

        # Make BinaryMap representations for each condition
        allowed_subs = {
            s for subs in df.var_wrt_ref
            for s in subs.split()
        }
        
        binmaps, X, y = {}, {}, {}
        for condition, condition_func_score_df in df.groupby("condition"):

            ref_bmap = bmap.BinaryMap(
                condition_func_score_df,
                substitutions_col="var_wrt_ref",
                allowed_subs=allowed_subs
            )
            binmaps[condition] = ref_bmap
            X[condition] = sparse.BCOO.from_scipy_sparse(ref_bmap.binary_variants)
            y[condition] = jnp.array(condition_func_score_df["func_score"].values)
        
        df.drop(["wts", "sites", "muts"], axis=1, inplace=True)

        return {'X':X, 'y':y}, binmaps, df, tuple(ref_bmap.all_subs), site_map 
