"""Tests for the Data class and its methods."""

import multidms
import numpy as np
import pandas as pd

func_score_df = pd.read_csv("tests/test_func_score.csv")
"""
condition aa_substitutions  func_score
a              M1E         2.0
a              G3R        -7.0
a              G3P        -0.5
a              M1W         2.3
b              M1E         1.0
b              P3R        -5.0
b              P3G         0.4
b          M1E P3G         2.7
b          M1E P3R        -2.7
b              P2T         0.3

   a  b
1  M  M
3  G  P
"""

# cond 1 G3P in ref
# cond 2 P3G in cond
# 1: cond1 and cond2
# 2: cond1 and !cond2

data = multidms.Data(
    func_score_df,
    alphabet=multidms.AAS_WITHSTOP,
    reference="a",
    assert_site_integrity=True,  # TODO testthis
)

model = multidms.Model(data, PRNGKey=23)


def test_bmap_mut_df_order():
    """
    Assert that the binarymap rows and columns match
    mutations_df indicies exactly.
    """
    # test the mutations order for both
    mut_df = data.mutations_df
    for condition in data.conditions:
        bmap = data.binarymaps[condition]
        assert np.all(mut_df.mutation == bmap.all_subs)

    # make sure the indices into the bmap are ordered 0-n
    for i, sub in enumerate(mut_df.mutation):
        assert sub == bmap.i_to_sub(i)


def test_converstion_from_subs():
    """Make sure that the conversion from each reference choice is correct"""
    for ref, bundle in zip(["a", "b"], ["G3P", "P3G"]):
        data = multidms.Data(func_score_df, reference=ref)
        assert data.convert_subs_wrt_ref_seq(("b" if ref == "a" else "a"), "") == bundle


def test_non_identical_conversion():
    """
    Test the conversion to with respect reference wt sequence.

    There are a few cases we will want to test:

    1. We drop sites (and the relevant variants with muts at those site)
    from the training data completely, if there is not information at
    a given site for all conditions. This is because we need to be able to
    infer the wildtypes for each condition in order to determine whether
    or not they should be treated as non-identical.

    2. Non identical "bundle muts" get encoded as 1
    in the non reference condition genotype

    3. Non identical site reversions don't exist
    in the non reference variants reference genotype
    """
    data = multidms.Data(
        func_score_df,
        alphabet=multidms.AAS_WITHSTOP,
        collapse_identical_variants="mean",
        reference="a",
        assert_site_integrity=True,
    )

    assert np.all(
        data.binarymaps["a"].binary_variants.todense()
        == [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]]
    )

    assert np.all(
        data.binarymaps["b"].binary_variants.todense()
        == [[1, 0, 1, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 1]]
    )


def test_model_PRNGKey():
    """
    Simply test the instatiation of a model with different PNRG keys
    to make sure the seed structure truly ensures the same parameter
    initialization values
    """
    model_1 = multidms.Model(data, PRNGKey=23)
    model_2 = multidms.Model(data, PRNGKey=23)
    for param, values in model_1.params.items():
        assert np.all(values == model_2.params[param])


def test_model_phenotype_predictions():
    """
    Make sure that the substitution conversion and binary
    encoding are correct in `Model.add_phenotype_to_df`
    by comparing the training data internal predictions
    match those of external predictions on that same data.
    """
    internal_pred = model.get_variants_df(phenotype_as_effect=False)
    external_pred = model.add_phenotypes_to_df(
        func_score_df, unknown_as_nan=True, phenotype_as_effect=False
    ).dropna()
    assert np.all(
        internal_pred.predicted_latent.values == external_pred.predicted_latent.values
    )
    assert np.all(
        internal_pred.predicted_func_score.values
        == external_pred.predicted_func_score.values
    )


def test_model_phenotype_effect_predictions():
    """
    Make sure that the substitution conversion and binary
    encoding are correct in `Model.add_phenotype_to_df`
    by comparing the training data internal predictions
    match those of external predictions on that same data.
    """
    internal_pred = model.get_variants_df(phenotype_as_effect=True)
    external_pred = model.add_phenotypes_to_df(
        func_score_df, unknown_as_nan=True, phenotype_as_effect=True
    ).dropna()
    assert np.all(
        internal_pred.predicted_latent.values == external_pred.predicted_latent.values
    )
    assert np.all(
        internal_pred.predicted_func_score.values
        == external_pred.predicted_func_score.values
    )
