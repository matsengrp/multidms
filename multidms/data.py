r"""
====
data
====

Defines :class:`Data` objects for handling data from one or more
dms experiments under various conditions.
"""

import os
from functools import partial
import warnings
import re

import binarymap as bmap
import numpy as onp
import pandas as pd
from polyclonal.plot import DEFAULT_POSITIVE_COLORS
from polyclonal.utils import MutationParser
from tqdm.auto import tqdm

from multidms import AAS

import jax
import jax.numpy as jnp
import seaborn as sns
from jax.experimental import sparse
from matplotlib import pyplot as plt
from pandarallel import pandarallel

jax.config.update("jax_enable_x64", True)


def split_sub(sub_string):
    """String match the wt, site, and sub aa
    in a given string denoting a single substitution
    """
    pattern = r"(?P<aawt>[A-Z])(?P<site>[\d\w]+)(?P<aamut>[A-Z\*])"
    match = re.search(pattern, sub_string)
    assert match is not None, sub_string
    return match.group("aawt"), str(match.group("site")), match.group("aamut")


def split_subs(subs_string, parser=split_sub):
    """Wrap the split_sub func to work for a
    string containing multiple substitutions
    """
    wts, sites, muts = [], [], []
    for sub in subs_string.split():
        wt, site, mut = parser(sub)
        wts.append(wt)
        sites.append(site)
        muts.append(mut)
    return wts, sites, muts


class Data:
    r"""
    Prep and store one-hot encoding of
    variant substitutions data.
    Individual objects of this type can be shared
    by multiple :py:class:`multidms.Model` Objects
    for efficiently fitting various models to the same data.

    Note
    ----
    You can initialize a :class:`Data` object with a :class:`pandas.DataFrame`
    with a row for each variant sampled and annotations
    provided in the required columns:

    1. `condition` - Experimental condition from
        which a sample measurement was obtained.
    2. `aa_substitutions` - Defines each variant
        :math:`v` as a string of substitutions (e.g., ``'M3A K5G'``).
        Note that while conditions may have differing wild types
        at a given site, the sites between conditions should reference
        the same site when alignment is performed between
        condition wild types.
    3. `func_score` - The functional score computed from experimental
        measurements.

    Parameters
    ----------
    variants_df : :class:`pandas.DataFrame` or None
        The variant level information from all experiments you
        wish to analyze. Should have columns named ``'condition'``,
        ``'aa_substitutions'``, and ``'func_score'``.
        See the class note for descriptions of each of the features.
    reference : str
        Name of the condition which annotates the reference.
        variants. Note that for model fitting this class will convert all
        amino acid substitutions for non-reference condition groups
        to relative to the reference condition.
        For example, if the wild type amino acid at site 30 is an
        A in the reference condition, and a G in a non-reference condition,
        then a Y30G mutation in the non-reference condition is recorded as an A30G
        mutation relative to the reference. This way, each condition informs
        the exact same parameters, even at sites that differ in wild type amino acid.
        These are encoded in a :class:`binarymap.binarymap.BinaryMap` object for each
        condition,
        where all sites that are non-identical to the reference are 1's.
        For motivation, see the `Model overview` section in :class:`multidms.Model`
        class notes.
    alphabet : array-like
        Allowed characters in mutation strings.
    collapse_identical_variants : {'mean', 'median', False}
        If identical variants in ``variants_df`` (same 'aa_substitutions'),
        exist within individual condition groups,
        collapse them by taking mean or median of 'func_score', or
        (if `False`) do not collapse at all. Collapsing will make fitting faster,
        but *not* a good idea if you are doing bootstrapping.
    condition_colors : array-like or dict
        Maps each condition to the color used for plotting. Either a dict keyed
        by each condition, or an array of colors that are sequentially assigned
        to the conditions.
    letter_suffixed_sites: bool
        True if sites are sequential and integer, False otherwise.
    assert_site_integrity : bool
        If True, will assert that all sites in the data frame
        have the same wild type amino acid, grouped by condition.
    verbose : bool
        If True, will print progress bars.
    nb_workers : int
        Number of workers to use for parallel operations.
        If None, will use all available CPUs.
    name : str or None
        Name of the data object. If None, will be assigned
        a unique name based upon the number of data objects
        instantiated.

    Example
    -------
    Simple example with two conditions (``'a'`` and ``'b'``)

    >>> import pandas as pd
    >>> import multidms
    >>> func_score_data = {
    ...     'condition' : ["a","a","a","a", "b","b","b","b","b","b"],
    ...     'aa_substitutions' : [
    ...         'M1E', 'G3R', 'G3P', 'M1W', 'M1E',
    ...         'P3R', 'P3G', 'M1E P3G', 'M1E P3R', 'P2T'
    ...     ],
    ...     'func_score' : [2, -7, -0.5, 2.3, 1, -5, 0.4, 2.7, -2.7, 0.3],
    ... }
    >>> func_score_df = pd.DataFrame(func_score_data)
    >>> func_score_df  # doctest: +NORMALIZE_WHITESPACE
    condition aa_substitutions  func_score
    0         a              M1E         2.0
    1         a              G3R        -7.0
    2         a              G3P        -0.5
    3         a              M1W         2.3
    4         b              M1E         1.0
    5         b              P3R        -5.0
    6         b              P3G         0.4
    7         b          M1E P3G         2.7
    8         b          M1E P3R        -2.7
    9         b              P2T         0.3

    Instantiate a ``Data`` Object allowing for stop codon variants
    and declaring condition `"a"` as the reference condition.

    >>> data = multidms.Data(
    ...     func_score_df,
    ...     alphabet = multidms.AAS_WITHSTOP,
    ...     reference = "a",
    ... )  # doctest: +ELLIPSIS
    ...

    Note this may take some time due to the string
    operations that must be performed when converting
    amino acid substitutions to be with respect to a
    reference wild type sequence.

    After the object has finished being instantiated,
    we now have access to a few 'static' properties
    of our data. See individual property docstring
    for more information.

    >>> data.reference
    'a'

    >>> data.conditions
    ('a', 'b')

    >>> data.mutations
    ('M1E', 'M1W', 'G3P', 'G3R')

    >>> data.site_map  # doctest: +NORMALIZE_WHITESPACE
    a  b
    1  M  M
    3  G  P

    >>> data.mutations_df  # doctest: +NORMALIZE_WHITESPACE
      mutation wts  sites muts  times_seen_a  times_seen_b
    0      M1E   M      1    E             1           3.0
    1      M1W   M      1    W             1           0.0
    2      G3P   G      3    P             1           1.0
    3      G3R   G      3    R             1           2.0

    >>> data.variants_df  # doctest: +NORMALIZE_WHITESPACE
      condition aa_substitutions  func_score var_wrt_ref
    0         a              M1E         2.0         M1E
    1         a              G3R        -7.0         G3R
    2         a              G3P        -0.5         G3P
    3         a              M1W         2.3         M1W
    4         b              M1E         1.0     G3P M1E
    5         b              P3R        -5.0         G3R
    6         b              P3G         0.4
    7         b          M1E P3G         2.7         M1E
    8         b          M1E P3R        -2.7     G3R M1E
    """

    counter = 0

    def __init__(
        self,
        variants_df: pd.DataFrame,
        reference: str,
        alphabet=AAS,
        collapse_identical_variants=False,
        condition_colors=DEFAULT_POSITIVE_COLORS,
        letter_suffixed_sites=False,
        assert_site_integrity=False,
        verbose=False,
        nb_workers=None,
        name=None,
    ):
        """See main class docstring."""
        # Check and initialize conditions attribute
        if pd.isnull(variants_df["condition"]).any():
            raise ValueError("condition name cannot be null")
        if variants_df["condition"].dtype.kind in "biufc":
            warnings.warn(
                "condition column looks to be numeric type, converting to string",
                UserWarning,
            )
        self._conditions = tuple(sorted(variants_df["condition"].astype(str).unique()))

        if str(reference) not in self._conditions:
            if not isinstance(reference, str):
                raise ValueError(
                    "reference must be a string, note that if your "
                    "condition names are numeric, they are being "
                    "converted to string"
                )
            raise ValueError("reference must be in condition factor levels")
        self._reference = str(reference)

        self._collapse_identical_variants = collapse_identical_variants

        # Check and initialize condition colors
        if isinstance(condition_colors, dict):
            self.condition_colors = {e: condition_colors[e] for e in self._conditions}
        elif len(condition_colors) < len(self._conditions):
            raise ValueError("not enough `condition_colors`")
        else:
            self.condition_colors = dict(zip(self._conditions, condition_colors))
        if not onp.all([isinstance(c, str) for c in self.condition_colors.values()]):
            raise ValueError("condition_color values must be hexadecimal")

        # Check and initialize alphabet & mut parser attributes
        if len(set(alphabet)) != len(alphabet):
            raise ValueError("duplicate letters in `alphabet`")
        self.alphabet = tuple(alphabet)

        # create mutation parser.
        self._mutparser = MutationParser(alphabet, letter_suffixed_sites)

        # Configure new variants df
        cols = ["condition", "aa_substitutions", "func_score"]
        if "weight" in variants_df.columns:
            cols.append(
                "weight"
            )  # will be overwritten if `self._collapse_identical_variants`
        if not variants_df[cols].notnull().all().all():
            raise ValueError(
                f"null entries in data frame of variants:\n{variants_df[cols]}"
            )

        # Create variants df attribute
        if self._collapse_identical_variants:
            agg_dict = {
                "weight": "sum",
                "func_score": self._collapse_identical_variants,
            }
            df = (
                variants_df[cols]
                .assign(weight=1)
                .groupby(["condition", "aa_substitutions"], as_index=False)
                .aggregate(agg_dict)
            )

        else:
            df = variants_df[cols].reset_index(drop=True)

        self._parse_muts = partial(split_subs, parser=self._mutparser.parse_mut)
        df["wts"], df["sites"], df["muts"] = zip(
            *df["aa_substitutions"].map(self._parse_muts)
        )

        # Use the "aa_substitutions" to infer the
        # wild type for each condition
        # site_map = pd.DataFrame()
        site_map = pd.DataFrame(columns=self.conditions)
        # print(site_map.info())
        for hom, hom_func_df in df.groupby("condition"):
            if verbose:
                print(f"inferring site map for {hom}")
            for idx, row in tqdm(
                hom_func_df.iterrows(), total=len(hom_func_df), disable=not verbose
            ):
                for wt, site in zip(row.wts, row.sites):
                    site_map.loc[site, hom] = wt

        if assert_site_integrity:
            if verbose:
                print("Asserting site integrity")
            for hom, hom_func_df in df.groupby("condition"):
                for idx, row in tqdm(
                    hom_func_df.iterrows(), total=len(hom_func_df), disable=not verbose
                ):
                    for wt, site in zip(row.wts, row.sites):
                        assert site_map.loc[site, hom] == wt

        # Throw variants if they contain non overlapping
        # mutations with all other conditions.
        na_rows = site_map.isna().any(axis=1)
        sites_to_throw = na_rows[na_rows].index
        site_map.dropna(inplace=True)

        nb_workers = min(os.cpu_count(), 4) if nb_workers is None else nb_workers
        pandarallel.initialize(
            progress_bar=verbose, verbose=0 if not verbose else 2, nb_workers=nb_workers
        )

        def flags_invalid_sites(disallowed_sites, sites_list):
            """Check to see if a sites list contains
            any disallowed sites
            """
            for site in sites_list:
                if site in disallowed_sites:
                    return False
            return True

        df["allowed_variant"] = df.sites.parallel_apply(
            lambda sl: flags_invalid_sites(sites_to_throw, sl)
        )
        if verbose:
            print(
                f"unknown cond wildtype at sites: {list(sites_to_throw.values)},"
                f"\ndropping: {len(df) - len(df[df['allowed_variant']])} variants"
                "which have mutations at those sites."
            )

        df.query("allowed_variant", inplace=True)
        df.drop("allowed_variant", axis=1, inplace=True)
        site_map.sort_index(inplace=True)

        def get_nis_from_site_map(site_map):
            """Get non-identical sites from a site map"""
            non_identical_sites = {}
            reference_sequence_conditions = [self._reference]
            for condition in self._conditions:
                if condition == self._reference:
                    non_identical_sites[condition] = []
                    continue

                nis = site_map.where(
                    site_map[self.reference] != site_map[condition],
                ).dropna()

                if len(nis) == 0:
                    non_identical_sites[condition] = []
                    reference_sequence_conditions.append(condition)
                else:
                    non_identical_sites[condition] = nis[[self._reference, condition]]
            return non_identical_sites, reference_sequence_conditions

        (non_identical_sites, reference_sequence_conditions) = get_nis_from_site_map(
            site_map
        )

        # invalid nis see https://github.com/matsengrp/multidms/issues/84
        observed_ref_muts = (
            df.query("condition == @self.reference")
            .aa_substitutions.str.split()
            .explode()
            .unique()
        )
        invalid_nim = []
        for condition in self.conditions:
            if (
                condition == self.reference
                or condition in reference_sequence_conditions
            ):
                continue
            observed_cond_muts = (
                df.query("condition == @condition")
                .aa_substitutions.str.split()
                .explode()
                .unique()
            )
            for site, cond_wts in non_identical_sites[condition].iterrows():
                ref_wt, cond_wt = cond_wts[self.reference], cond_wts[condition]
                forward_mut = f"{ref_wt}{site}{cond_wt}"
                reversion_mut = f"{cond_wt}{site}{ref_wt}"

                condition_1 = forward_mut in observed_ref_muts
                condition_2 = reversion_mut in observed_cond_muts
                if not (condition_1 and condition_2):
                    invalid_nim.append(site)

        # find variants that contain mutations at invalid sites
        df["allowed_variant"] = df.sites.parallel_apply(
            lambda sl: flags_invalid_sites(invalid_nim, sl)
        )
        if verbose:
            print(
                f"invalid non-identical-sites: {invalid_nim}, dropping "
                f"{len(df) - len(df[df['allowed_variant']])} variants"
            )

        # drop variants that contain mutations at invalid sites
        df.query("allowed_variant", inplace=True)
        df.drop("allowed_variant", axis=1, inplace=True)

        # drop invalid sites from site map
        self._site_map = site_map.drop(invalid_nim, inplace=False)

        # recompute non-identical sites for static property
        (
            self._non_identical_sites,
            self._reference_sequence_conditions,
        ) = get_nis_from_site_map(self._site_map)

        # compute the static non_identical_mutations property
        non_identical_mutations = {}
        for condition in self.conditions:
            if condition in self.reference_sequence_conditions:
                non_identical_mutations[condition] = ""
                continue
            nis = self.non_identical_sites[condition]
            muts = nis[self.reference] + nis.index.astype(str) + nis[condition]
            muts_string = " ".join(muts.values)
            non_identical_mutations[condition] = muts_string
        self._non_identical_mutations = non_identical_mutations

        # compute all substitution conversions for all conditions which
        # do not share the reference sequence
        df = df.assign(var_wrt_ref=df["aa_substitutions"])
        for condition, condition_func_df in df.groupby("condition"):
            if verbose:
                print(f"Converting mutations for {condition}")
            if condition in self.reference_sequence_conditions:
                if verbose:
                    print("is reference, skipping")
                continue

            idx = condition_func_df.index
            df.loc[idx, "var_wrt_ref"] = condition_func_df.parallel_apply(
                lambda x: self._convert_split_subs_wrt_ref_seq(
                    condition, x.wts, x.sites, x.muts
                ),
                axis=1,
            )

        # Make BinaryMap representations for each condition
        allowed_subs = {s for subs in df.var_wrt_ref for s in subs.split()}
        binmaps, X, y = {}, {}, {}
        for condition, condition_func_score_df in df.groupby("condition"):
            ref_bmap = bmap.BinaryMap(
                condition_func_score_df,
                substitutions_col="var_wrt_ref",
                allowed_subs=allowed_subs,
                alphabet=self.alphabet,
                sites_as_str=letter_suffixed_sites,
            )
            binmaps[condition] = ref_bmap
            X[condition] = sparse.BCOO.from_scipy_sparse(ref_bmap.binary_variants)
            y[condition] = jnp.array(condition_func_score_df["func_score"].values)

        df.drop(["wts", "sites", "muts"], axis=1, inplace=True)
        self._variants_df = df
        self._training_data = {"X": X, "y": y}
        self._binarymaps = binmaps
        self._mutations = tuple(ref_bmap.all_subs)

        # initialize single mutational effects df
        mut_df = pd.DataFrame({"mutation": self._mutations})

        mut_df["wts"], mut_df["sites"], mut_df["muts"] = zip(
            *mut_df["mutation"].map(self._mutparser.parse_mut)
        )

        # compute times seen in data
        for condition, condition_vars in self._variants_df.groupby("condition"):
            times_seen = (
                condition_vars["var_wrt_ref"].str.split().explode().value_counts()
            )
            if (times_seen == times_seen.astype(int)).all():
                times_seen = times_seen.astype(int)
            times_seen.index.name = "mutation"
            times_seen.name = f"times_seen_{condition}"
            mut_df = mut_df.merge(times_seen, on="mutation", how="left").fillna(0)

        self._mutations_df = mut_df
        self._name = name if isinstance(name, str) else f"Data-{Data.counter}"
        Data.counter += 1

    def __repr__(self):
        """Returns a string representation of the object."""
        return f"{self.__class__.__name__}({self.name})"

    def _str__(self):
        """Returns a string representation of the object."""
        return f"{self.__class__.__name__}({self.name})"

    @property
    def name(self) -> str:
        """The name of the data object."""
        return self._name

    @property
    def conditions(self) -> tuple:
        """A tuple of all conditions."""
        return self._conditions

    @property
    def reference(self) -> str:
        """The name of the reference condition."""
        return self._reference

    @property
    def mutations(self) -> tuple:
        """
        A tuple of all mutations in the order relative to their index into
        the binarymap.
        """
        return self._mutations

    @property
    def mutations_df(self) -> pd.DataFrame:
        """A dataframe summarizing all single mutations"""
        return self._mutations_df

    @property
    def variants_df(self) -> pd.DataFrame:
        """A dataframe summarizing all variants in the training data."""
        return self._variants_df

    @property
    def site_map(self) -> pd.DataFrame:
        """
        A dataframe indexed by site, with columns
        for all conditions giving the wild type amino acid
        at each site.
        """
        return self._site_map

    @property
    def non_identical_mutations(self) -> dict:
        """
        A dictionary keyed by condition names with values
        being a string of all mutations that differ from the
        reference sequence.
        """
        return self._non_identical_mutations

    @property
    def non_identical_sites(self) -> dict:
        """
        A dictionary keyed by condition names with values
        being a :class:`pandas.DataFrame` indexed by site,
        with columns for the reference
        and non-reference amino acid at each site that differs.
        """
        return self._non_identical_sites

    @property
    def reference_sequence_conditions(self) -> list:
        """
        A list of conditions that have the same wild type
        sequence as the reference condition.
        """
        return self._reference_sequence_conditions

    @property
    def training_data(self) -> dict:
        """A dictionary with keys 'X' and 'y' for the training data."""
        return self._training_data

    @property
    def binarymaps(self) -> dict:
        """
        A dictionary keyed by condition names with values
        being a ``BinaryMap`` object for each condition.
        """
        return self._binarymaps

    @property
    def targets(self) -> dict:
        """The functional scores for each variant in the training data."""
        return self._training_data["y"]

    # TODO, rename mutparser
    @property
    def mutparser(self) -> MutationParser:
        """
        The mutation ``polyclonal.utils.MutationParser`` used
        to parse mutations.
        """
        return self._mutparser

    # TODO, rename
    @property
    def parse_mut(self) -> MutationParser:
        """
        returns a function that splits a single amino acid substitutions
        into wildtype, site, and mutation
        using the mutation parser.
        """
        return self.mutparser.parse_mut

    # TODO, document rename issue
    @property
    def parse_muts(self) -> partial:
        """
        A function that splits amino acid substitutions
        (a string of more than one) into wildtype, site, and mutation
        using the mutation parser.
        """
        return self._parse_muts

    # TODO should this be cached? how does caching interact with the way in
    # which we applying this function in parallel?
    # although, unless the variants are un-collapsed, this cache will be
    # pretty useless.
    # although it could be useful for the Model.add_phenotypes_to_df method.
    def convert_subs_wrt_ref_seq(self, condition, aa_subs):
        """
        Covert amino acid substitutions to be with respect to the reference sequence.

        Parameters
        ----------
        condition : str
            The condition from which aa substitutions are relative to.
        aa_subs : str
            A string of amino acid substitutions, relative to the condition sequence,
            to converted

        Returns
        -------
        str
            A string of amino acid substitutions relative to the reference sequence.
        """
        if condition not in self.conditions:
            raise ValueError(f"condition {condition} does not exist in model")
        if condition in self.reference_sequence_conditions:
            return aa_subs
        return self._convert_split_subs_wrt_ref_seq(
            condition, *self.parse_muts(aa_subs)
        )

    def _convert_split_subs_wrt_ref_seq(self, condition, wts, sites, muts):
        """
        Covert amino acid substitutions to be with respect to the reference sequence.

        Parameters
        ----------
        condition : str
            The condition from which aa substitutions are relative to.
        wts : array-like
            The wild type amino acids for each substitution.
        sites : array-like
            The sites for each substitution.
        muts : array-like
            The mutant amino acids for each substitution.

        Returns
        -------
        str
            A string of amino acid substitutions relative to the reference sequence.
        """
        assert len(wts) == len(sites) == len(muts)

        nis = self.non_identical_sites[condition]
        ret = self.non_identical_sites[condition].copy()

        for wt, site, mut in zip(wts, sites, muts):
            if site not in nis.index.values:
                ret.loc[site] = wt, mut
            else:
                ref_wt = nis.loc[site, self.reference]
                if mut != ref_wt:
                    ret.loc[site] = ref_wt, mut
                else:
                    ret.drop(site, inplace=True)

        converted_muts = ret[self.reference] + ret.index.astype(str) + ret[condition]
        return " ".join(converted_muts)

    def plot_times_seen_hist(self, saveas=None, show=True, **kwargs):
        """Plot a histogram of the number of times each mutation was seen."""
        times_seen_cols = [f"times_seen_{c}" for c in self._conditions]
        fig, ax = plt.subplots()
        sns.histplot(self._mutations_df[times_seen_cols], ax=ax, **kwargs)
        if saveas:
            fig.saveas(saveas)
        if show:
            plt.show()
        return fig, ax

    def plot_func_score_boxplot(self, saveas=None, show=True, **kwargs):
        """Plot a boxplot of the functional scores for each condition."""
        fig, ax = plt.subplots()
        sns.boxplot(
            self._variants_df,
            x="condition",
            y="func_score",
            ax=ax,
            notch=True,
            showcaps=False,
            flierprops={"marker": "x"},
            boxprops={"facecolor": (0.4, 0.6, 0.8, 0.5)},
            medianprops={"color": "coral"},
            **kwargs,
        )

        if saveas:
            fig.saveas(saveas)
        if show:
            plt.show()
        return fig, ax
