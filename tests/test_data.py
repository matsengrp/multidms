"""Tests for the Data class and its methods."""


import os
import pytest
import multidms
import numpy as np
import pandas as pd
from io import StringIO

TEST_FUNC_SCORES = pd.read_csv(
    StringIO(
        """
condition,aa_substitutions,func_score
a,M1E,2.0
a,G3R,-7.0
a,G3P,-0.5
a,M1W,2.3
b,M1E,1.0
b,P3R,-5.0
b,P3G,0.4
b,M1E P3G,2.7
b,M1E P3R,-2.7
b,P2T,0.3
        """
    )
)

data = multidms.Data(
    TEST_FUNC_SCORES,
    alphabet=multidms.AAS_WITHSTOP,
    reference="a",
    assert_site_integrity=True,
)

model = multidms.Model(data, PRNGKey=23)


def test_site_integrity():
    """
    Test that the site integrity is maintained
    by raising a ValueError if it is not when using assert_site_integrity=True
    """
    df = pd.concat(
        [
            TEST_FUNC_SCORES,
            pd.Series({"condition": "a", "aa_substitutions": "P2E", "func_score": 0.0}),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError):
        multidms.Data(
            df,
            alphabet=multidms.AAS_WITHSTOP,
            reference="a",
            assert_site_integrity=True,
        )


def test_bmap_mut_df_order():
    """
    Assert that the binarymap rows and columns match
    mutations_df indicies exactly.
    """
    mut_df = data.mutations_df
    for condition in data.conditions:
        bmap = data.binarymaps[condition]
        assert np.all(mut_df.mutation == bmap.all_subs)

    # make sure the indices into the bmap are ordered 0-n
    for i, sub in enumerate(mut_df.mutation):
        assert sub == bmap.i_to_sub(i)


def test_non_identical_mutations():
    """
    Test that the non identical mutations
    are correctly identified.
    """
    data = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=False,
    )
    assert data.non_identical_mutations["a"] == ""
    assert data.non_identical_mutations["b"] == "G3P"

    data = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="b",
        assert_site_integrity=True,
    )
    assert data.non_identical_mutations["a"] == "P3G"
    assert data.non_identical_mutations["b"] == ""


def test_invalid_non_identical_sites():
    """
    test that data throws non-identical sites,
    and related variants, when we don't have
    'forward' and 'reverse' mutational information
    as discussed in https://github.com/matsengrp/multidms/issues/84.
    """
    # same data but dropped the reversion mut for condition b
    data_no_forward = "not aa_substitutions.str.contains('P3G')"
    data_no_reversion = "aa_substitutions != 'G3P'"
    data_neither = f"{data_no_forward} & {data_no_reversion}"
    # we expect now, that the only variants kept should be those
    # that only contain exactly a mutation at site 1, there's three of those
    for query in [data_no_forward, data_no_reversion, data_neither]:
        data = multidms.Data(
            TEST_FUNC_SCORES.query(query),
            alphabet=multidms.AAS_WITHSTOP,
            reference="a",
            assert_site_integrity=True,
        )
        assert len(data.variants_df) == 3
        assert len(data.non_identical_mutations["a"]) == 0
        assert len(data.non_identical_mutations["b"]) == 0
        assert len(data.non_identical_sites["a"]) == 0
        assert len(data.non_identical_sites["b"]) == 0
        assert data.reference_sequence_conditions == ["a", "b"]


def test_converstion_from_subs():
    """Make sure that the conversion from each reference choice is correct"""
    for ref, bundle in zip(["a", "b"], ["G3P", "P3G"]):
        data = multidms.Data(TEST_FUNC_SCORES, reference=ref)
        assert data.convert_subs_wrt_ref_seq(("b" if ref == "a" else "a"), "") == bundle


def test_wildtype_mutatant_predictions():
    """
    test that the wildtype predictions are correct
    by comparing them to a "by-hand" calculation on the parameters.
    """
    data = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=False,
    )
    model = multidms.Model(data, PRNGKey=23)
    model.fit(maxiter=2)
    wildtype_df = model.wildtype_df
    for condition in model.data.conditions:
        byhand_latent = model.params["beta_naught"][0]
        if condition != model.data.reference:
            converted_subs = model.data.convert_subs_wrt_ref_seq(condition, "")
            bmap = model.data.binarymaps[model.data.reference]
            enc = bmap.sub_str_to_binary(converted_subs)
            assert sum(enc) == len(converted_subs.split())
            mut_params = model.get_mutations_df(phenotype_as_effect=False).query(
                "mutation.isin(@converted_subs.split())"
            )
            latent = (mut_params.beta + mut_params[f"shift_{condition}"]).sum()
            offset = model.params["beta_naught"] + model.params[f"alpha_{condition}"]
            byhand_latent = latent + offset[0]

        # check latent
        pred_latent = wildtype_df.loc[condition, "predicted_latent"]
        assert np.isclose(byhand_latent, pred_latent)

        # check wt functional score
        sig_params = model.params["theta"]
        scale, bias = sig_params["ge_scale"], sig_params["ge_bias"]
        byhand_func_score = scale / (1 + np.exp(-1 * byhand_latent)) + bias
        pred_func_score = wildtype_df.loc[condition, "predicted_func_score"]
        assert np.isclose(byhand_func_score, pred_func_score)


def test_mutations_df():
    """
    make sure that the functional score predictions
    for individual mutations is correct by comparing them to by-hand
    calculations.
    """
    data = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=False,
    )
    model = multidms.Model(data, PRNGKey=23)
    model.fit(maxiter=2)
    sig_params = model.params["theta"]
    scale, bias = sig_params["ge_scale"], sig_params["ge_bias"]
    mutations_df = model.get_mutations_df(phenotype_as_effect=False)
    for i, mutation in enumerate(model.data.mutations):
        effect = model.params["beta"][i]
        for condition in model.data.conditions:
            shift = model.params[f"shift_{condition}"][i]
            byhand_latent = (
                effect
                + shift
                + model.params["beta_naught"]
                + model.params[f"alpha_{condition}"]
            )
            byhand_func_score = scale / (1 + np.exp(-1 * byhand_latent)) + bias
            assert np.isclose(
                byhand_func_score,
                mutations_df.loc[mutation, f"predicted_func_score_{condition}"],
            )


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
        TEST_FUNC_SCORES,
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


def test_lower_bound():
    """
    Make sure that the softplus lower bound is correct
    by initializing a sofplus activation models and asserting
    predictions never go below the specified lower bound.
    even if that lower bound is high!

    Note that while the lower bound is set, the
    "effective phenotype" lower bound is
    equal to the lower bound - the wildtype phenotype
    for any given condition. However, since the latent
    offset parameters are initialized to 0.0 by default,
    the reference wildtype phenotype is 0.0 before fitting.
    """
    model = multidms.Model(
        data,
        output_activation=multidms.biophysical.softplus_activation,
        PRNGKey=23,
        lower_bound=1000.0,
    )
    variants_df = model.get_variants_df(phenotype_as_effect=False)
    assert np.all(variants_df.predicted_func_score >= 1000.0)


def test_null_post_latent():
    """
    Make sure that setting the epistatic model, and output
    activation to the identity, results in the same predictions
    as the the additive model.
    """
    model = multidms.Model(
        data,
        epistatic_model=multidms.biophysical.identity_activation,
        output_activation=multidms.biophysical.identity_activation,
        PRNGKey=23,
    )
    variants_df = model.get_variants_df(phenotype_as_effect=False)
    assert np.all(variants_df.predicted_latent == variants_df.predicted_func_score)


def test_model_phenotype_predictions():
    """
    Make sure that the substitution conversion and binary
    encoding are correct in `Model.add_phenotype_to_df`
    by comparing the training data internal predictions
    match those of external predictions on that same data.
    """
    internal_pred = model.get_variants_df(phenotype_as_effect=False)
    external_pred = model.add_phenotypes_to_df(
        TEST_FUNC_SCORES, unknown_as_nan=True, phenotype_as_effect=False
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
        TEST_FUNC_SCORES, unknown_as_nan=True, phenotype_as_effect=True
    ).dropna()
    assert np.all(
        internal_pred.predicted_latent.values == external_pred.predicted_latent.values
    )
    assert np.all(
        internal_pred.predicted_func_score.values
        == external_pred.predicted_func_score.values
    )


def test_model_fit_and_determinism():
    """
    Make sure that the model is deterministic by fitting
    the model twice and making sure that the parameters
    are the same.
    """
    model_1 = multidms.Model(data, PRNGKey=23)
    model_2 = multidms.Model(data, PRNGKey=23)

    model_1.fit(maxiter=5)
    model_2.fit(maxiter=5)

    for param, values in model_1.params.items():
        assert np.all(values == model_2.params[param])


def test_explode_params_dict():
    """Test multidms.model_collection.explode_params_dict"""
    params_dict = {"a": [1, 2], "b": [3]}
    exploded = multidms.model_collection._explode_params_dict(params_dict)
    assert exploded == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def test_fit_models():
    """
    Test fitting two different models in
    parallel using multidms.model_collection.fit_models
    """
    data = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=False,
    )
    params = {
        "dataset": [data],
        "iterations_per_step": [2],
        "scale_coeff_lasso_shift": [0.0, 1e-5],
    }
    _, _, fit_models_df = multidms.model_collection.fit_models(
        params,
        n_threads=min(os.cpu_count(), 4),
    )
    mc = multidms.model_collection.ModelCollection(fit_models_df)
    tall_combined = mc.split_apply_combine_muts(groupby=("scale_coeff_lasso_shift"))
    assert len(tall_combined) == 2 * len(data.mutations_df)
    assert list(tall_combined.index.names) == ["scale_coeff_lasso_shift"]


def test_data_names():
    """
    Test that the default data names are correctly
    updated as new Data objects are created within a
    single python session.
    """
    num_datasets = multidms.Data.counter
    for i in range(3):
        d_i = multidms.Data(
            TEST_FUNC_SCORES,
            alphabet=multidms.AAS_WITHSTOP,
            reference="a",
            assert_site_integrity=False,
        )
        assert d_i.name == f"Data-{num_datasets + i}"
