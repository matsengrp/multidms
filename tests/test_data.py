"""Tests for the Data class and its methods."""

import pytest
import multidms
import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_flatten
import pandas as pd
from io import StringIO

import multidms.utils

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
    name="test_data",
    include_counts=False,
)
model = multidms.Model(data, PRNGKey=23)


r"""
+++++++++++++++++++++++++++++
DATA
+++++++++++++++++++++++++++++
"""


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
    mutations_df indices exactly.
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
        include_counts=False,
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
    Test that data throws non-identical sites,
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


def test_conversion_from_subs():
    """Make sure that the conversion from each reference choice is correct"""
    for ref, bundle in zip(["a", "b"], ["G3P", "P3G"]):
        data = multidms.Data(TEST_FUNC_SCORES, reference=ref)
        assert data.convert_subs_wrt_ref_seq(("b" if ref == "a" else "a"), "") == bundle


def test_non_identical_mutations_property():
    """Make sure we're getting the correct indicies for the bundle mutations"""
    assert jnp.all(data.bundle_idxs["a"] == jnp.repeat(False, len(data.mutations)))
    assert jnp.sum(data.bundle_idxs["b"]) == 1


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


def test_single_mut_encodings():
    """
    Test that the binary encoding of single mutations
    is correct.
    """
    data = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=False,
        include_counts=False,
    )
    single_mut_encodings = data.single_mut_encodings
    assert np.all(
        np.array(single_mut_encodings["a"].todense()).flatten()
        == np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).flatten()
    )
    assert np.all(
        np.array(single_mut_encodings["b"].todense()).flatten()
        == np.array([[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).flatten()
    )


def test_plotting_fxns():
    """Test that the plotting functions work"""
    Data = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=False,
    )

    Data.plot_times_seen_hist(show=False)
    Data.plot_func_score_boxplot(show=False)


r"""
+++++++++++++++++++++++++++++
UTILS
+++++++++++++++++++++++++++++
"""


def test_explode_params_dict():
    """Test multidms.model_collection.explode_params_dict"""
    params_dict = {"a": [1, 2], "b": [3]}
    exploded = multidms.utils.explode_params_dict(params_dict)
    assert exploded == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def test_difference_matrix():
    """
    Test that the difference matrix performs
    the correct linear operation
    """
    # test difference matrix at 5 different n's
    for n in range(1, 5):
        test_vec = jnp.array(range(1, n + 1))[:, None]
        expected_result = jnp.array(range(n))[:, None]
        D = multidms.utils.difference_matrix(n)
        assert jnp.all(D @ test_vec == expected_result)


def test_transform_inverse():
    """
    Test that the transform operation is
    its own inverse.
    """
    model = multidms.Model(data, multidms.biophysical.identity_activation, PRNGKey=23)
    params = model._scaled_data_params
    bundle_idxs = data.bundle_idxs
    double_inverse = multidms.utils.transform(
        multidms.utils.transform(params, bundle_idxs), bundle_idxs
    )
    assert np.all(params["beta"]["a"] == double_inverse["beta"]["a"])
    assert np.all(params["beta"]["b"] == double_inverse["beta"]["b"])
    assert np.all(params["beta0"]["a"] == double_inverse["beta0"]["a"])
    assert np.all(params["beta0"]["b"] == double_inverse["beta0"]["b"])


r"""
+++++++++++++++++++++++++++++
MODEL
+++++++++++++++++++++++++++++
"""


def test_linear_model_fit_simple():
    """
    Simple test to see that the linear model
    fits without error.
    """
    data = multidms.Data(
        TEST_FUNC_SCORES.query("condition == 'a'"),
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=False,
        include_counts=False,
    )
    model = multidms.Model(data, multidms.biophysical.identity_activation, PRNGKey=23)
    model.fit(maxiter=2, warn_unconverged=False)

    # test all plotting fxn's
    model.plot_pred_accuracy(show=False)
    model.plot_epistasis(show=False)
    model.plot_param_hist("beta_a", show=False)
    model.plot_param_heatmap("beta_a", show=False)
    _ = model.mut_param_heatmap("beta")


def test_linear_model_multi_cond_fit_simple():
    """
    Simple test to see that the linear model
    fits multiple conditions without error.
    """
    data = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=False,
        include_counts=False,
    )
    assert np.all([not bi for bi in list(data.bundle_idxs["a"])])
    model = multidms.Model(data, multidms.biophysical.identity_activation, PRNGKey=23)
    model.fit(maxiter=2, warn_unconverged=False)

    # test all plotting fxn's
    model.plot_pred_accuracy(show=False)
    model.plot_epistasis(show=False)
    model.plot_param_hist("shift_b", show=False)
    model.plot_param_heatmap("shift_b", show=False)
    model.plot_shifts_by_site("b", show=False)
    _ = model.mut_param_heatmap("shift")


def test_fit_simple():
    """
    Simple test to see that the single-condition model
    fits without error.
    """
    data = multidms.Data(
        TEST_FUNC_SCORES.query("condition == 'a'"),
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=False,
        include_counts=False,
    )
    model = multidms.Model(data, PRNGKey=23)
    loss = model.loss
    model.fit(maxiter=2, warn_unconverged=False)
    assert loss != model.loss

    # test all plotting fxn's
    model.plot_pred_accuracy(show=False)
    model.plot_epistasis(show=False)
    model.plot_param_hist("beta_a", show=False)
    model.plot_param_heatmap("beta_a", show=False)
    _ = model.mut_param_heatmap("beta")


def test_multi_cond_fit_simple():
    """
    Simple test to make sure the multi-condition model
    fits without error.
    """
    data = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=False,
        include_counts=False,
    )
    model = multidms.Model(data, PRNGKey=23)
    model.fit(maxiter=2, warn_unconverged=False)

    # test all plotting fxn's
    model.plot_pred_accuracy(show=False)
    model.plot_epistasis(show=False)
    model.plot_param_hist("shift_b", show=False)
    model.plot_param_heatmap("shift_b", show=False)
    model.plot_shifts_by_site("b", show=False)
    _ = model.mut_param_heatmap()


def test_scaled_predictions():
    """
    Test that the scaled data and parameter predictions
    are the same as unscaled predictions.
    """
    model = multidms.Model(data, PRNGKey=23)
    model.fit(maxiter=2, warn_unconverged=False)
    pred_fxn = model.model_components["f"]
    scaled_params = model._scaled_data_params
    scaled_data = model.data.scaled_training_data["X"]
    unscaled_params = model.params
    unscaled_data = model.data.training_data["X"]
    for condition in model.data.conditions:
        scaled_d_params = {
            "beta": scaled_params["beta"][condition],
            "beta0": scaled_params["beta0"][condition],
            "theta": scaled_params["theta"],
        }
        scaled_d_data = scaled_data[condition]
        scaled_predictions = pred_fxn(scaled_d_params, scaled_d_data)

        unscaled_d_params = {
            "beta": unscaled_params["beta"][condition],
            "beta0": unscaled_params["beta0"][condition],
            "theta": unscaled_params["theta"],
        }
        unscaled_d_data = unscaled_data[condition]
        unscaled_predictions = pred_fxn(unscaled_d_params, unscaled_d_data)

        assert np.all(scaled_predictions == unscaled_predictions)


def test_wildtype_mutant_predictions():
    """
    Test that the wildtype predictions are correct
    by comparing them to a "by-hand" calculation on the parameters.
    """
    data = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=False,
        include_counts=False,
    )
    model = multidms.Model(data, PRNGKey=23)
    model.fit(maxiter=2, warn_unconverged=False)
    wildtype_df = model.wildtype_df
    for condition in model.data.conditions:
        latent_offset = model.params["beta0"][condition]
        byhand_wt_latent_pred = latent_offset

        if condition != model.data.reference:
            converted_subs = model.data.convert_subs_wrt_ref_seq(condition, "")
            bmap = model.data.binarymaps[model.data.reference]
            enc = bmap.sub_str_to_binary(converted_subs)
            assert sum(enc) == len(converted_subs.split())
            mut_params = model.get_mutations_df().query(
                "mutation.isin(@converted_subs.split())"
            )
            bundle_effect = mut_params[f"beta_{condition}"].sum()

            # first, check that the bundle effect of conditional beta's
            # is the same as the reference beta's plus the shift
            reference_beta = mut_params[f"beta_{model.data.reference}"]
            conditional_shift = mut_params[f"shift_{condition}"]
            bundle_effect_beta_shift = (reference_beta + conditional_shift).sum()
            assert np.isclose(bundle_effect, bundle_effect_beta_shift)

            byhand_wt_latent_pred += bundle_effect

        # check latent
        method_wt_latent_pred = wildtype_df.loc[condition, "predicted_latent"]
        assert np.isclose(byhand_wt_latent_pred, method_wt_latent_pred)

        # check wt functional score
        sig_params = model.params["theta"]
        scale, bias = sig_params["ge_scale"], sig_params["ge_bias"]
        byhand_func_score = scale / (1 + np.exp(-1 * byhand_wt_latent_pred)) + bias
        pred_func_score = wildtype_df.loc[condition, "predicted_func_score"]
        assert np.isclose(byhand_func_score, pred_func_score)


def test_mutations_df():
    """
    Make sure that the functional score predictions
    for individual mutations is correct by comparing them to by-hand
    calculations.
    """
    data = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=False,
        include_counts=False,
    )
    model = multidms.Model(data, PRNGKey=23)
    model.fit(maxiter=2, warn_unconverged=False)

    # We want to make sure the predictions in this method are as expected
    mutations_df = model.get_mutations_df()

    # we'll compare it to predictions done by hand
    sig_params = model.params["theta"]
    scale, bias = sig_params["ge_scale"][0], sig_params["ge_bias"][0]
    wildtype_df = model.wildtype_df

    for condition in model.data.conditions:
        wildtype_func_score = wildtype_df.loc[condition, "predicted_func_score"]
        for i, mutation in enumerate(model.data.mutations):
            mut_df_pred = mutations_df.loc[
                mutation, f"predicted_func_score_{condition}"
            ]

            ref_effect = model.params["beta"][data.reference][i]
            shift = model.params["shift"][f"{condition}"][i]
            effect = ref_effect + shift
            assert np.isclose(effect, model.params["beta"][condition][i])

            converted_wrt_ref = model.data.convert_subs_wrt_ref_seq(condition, mutation)
            binarymap = model.data.binarymaps[condition]
            converted_wrt_ref_enc = binarymap.sub_str_to_binary(converted_wrt_ref)
            bool_enc = converted_wrt_ref_enc.astype(bool)
            additive_effect = sum(model.params["beta"][condition][bool_enc])
            latent_offset = model.params["beta0"][condition][0]
            byhand_latent = additive_effect + latent_offset
            byhand_func_score = scale / (1 + np.exp(-1 * byhand_latent)) + bias
            byhand_func_score_effect = byhand_func_score - wildtype_func_score

            assert np.isclose(
                byhand_func_score_effect,
                mut_df_pred,
            )


def test_model_PRNGKey():
    """
    Simply test the instantiation of a model with different PRNG keys
    to make sure the seed structure truly ensures the same parameter
    initialization values. Note the only random initialization in the
    models is with the Neural network non-linearity model.
    """
    model_1 = multidms.Model(
        data, epistatic_model=multidms.biophysical.nn_global_epistasis, PRNGKey=23
    )
    model_2 = multidms.Model(
        data, epistatic_model=multidms.biophysical.nn_global_epistasis, PRNGKey=23
    )
    assert tree_flatten(model_1.params)[1] == tree_flatten(model_2.params)[1]


def test_lower_bound():
    """
    Make sure that the softplus lower bound is correct
    by initializing a softplus activation models and asserting
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
    assert np.allclose(
        internal_pred.predicted_latent.values, external_pred.predicted_latent.values
    )
    assert np.allclose(
        internal_pred.predicted_func_score.values,
        external_pred.predicted_func_score.values,
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
    assert np.allclose(
        internal_pred.predicted_latent.values, external_pred.predicted_latent.values
    )
    assert np.allclose(
        internal_pred.predicted_func_score.values,
        external_pred.predicted_func_score.values,
    )


def test_model_fit_and_determinism():
    """
    Make sure that the model is deterministic by fitting
    the model twice and making sure that the parameters
    are the same.
    """
    model_1 = multidms.Model(data, PRNGKey=23)
    model_2 = multidms.Model(data, PRNGKey=23)

    model_1.fit(maxiter=5, warn_unconverged=False)
    model_2.fit(maxiter=5, warn_unconverged=False)
    assert tree_flatten(model_1.params)[1] == tree_flatten(model_2.params)[1]


def test_model_get_df_loss():
    """
    Test that the loss is correctly calculated
    by comparing the result of model.loss() to the results of model.get_df_loss()
    when given the training dataframe.
    """
    model = multidms.Model(data, PRNGKey=23)
    model.fit(maxiter=2, warn_unconverged=False)
    loss = model.loss
    df_loss = model.get_df_loss(TEST_FUNC_SCORES)
    assert loss == df_loss

    # also test that is it's the same if we add an unknown variant to training
    test_with_unknown = TEST_FUNC_SCORES.copy()
    test_with_unknown.loc[len(test_with_unknown)] = ["a", "E100T MIE", 0.2]
    df_loss = model.get_df_loss(test_with_unknown)
    assert loss == df_loss


def test_model_get_df_loss_conditional():
    """
    Test that the loss is correctly calculated
    across each condition, by summing the conditions to be sure
    they match the total loss.
    """
    model = multidms.Model(data, PRNGKey=23)
    model.fit(maxiter=2, warn_unconverged=False)
    loss = model.loss
    df_loss = model.get_df_loss(TEST_FUNC_SCORES, conditional=True)
    # remove full and compare sum of the rest
    df_loss.pop("total")
    assert loss == sum(df_loss.values()) / len(df_loss)


def test_conditional_loss():
    """
    Test that the conditional loss is correctly calculated
    by comparing the result of model.conditional_loss()
    to the results of model.get_df_loss()
    when given the training dataframe.
    """
    model = multidms.Model(data, PRNGKey=23)
    model.fit(maxiter=2, warn_unconverged=False)
    loss = model.conditional_loss
    df_loss = model.get_df_loss(TEST_FUNC_SCORES, conditional=True)
    assert loss == df_loss


r"""
+++++++++++++++++++++++++++++
MODEL_COLLECTION
+++++++++++++++++++++++++++++
"""


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
        include_counts=False,
    )
    params = {
        "dataset": [data],
        "maxiter": [2],
        "scale_coeff_lasso_shift": [0.0, 1e-5],
    }
    _, _, fit_models_df = multidms.model_collection.fit_models(
        params,
        n_threads=-1,
    )
    # assert False
    mc = multidms.model_collection.ModelCollection(fit_models_df)
    tall_combined = mc.split_apply_combine_muts(groupby=("scale_coeff_lasso_shift"))
    assert len(tall_combined) == 2 * len(data.mutations_df)
    assert list(tall_combined.index.names) == ["scale_coeff_lasso_shift"]


def test_ModelCollection_mut_param_dataset_correlation():
    """
    Test that the correlation between the mutational
    parameter estimates across conditions is correct.
    by correlating two deterministic model fits from identical
    datasets, meaning they should have a correlation of 1.0.
    """
    data_rep1 = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=False,
        name="rep1",
        include_counts=False,
    )

    data_rep2 = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=False,
        name="rep2",
        include_counts=False,
    )

    params = {
        "dataset": [data_rep1, data_rep2],
        "maxiter": [1],
        "scale_coeff_lasso_shift": [0.0],
    }

    _, _, fit_models_df = multidms.model_collection.fit_models(
        params,
        n_threads=-1,
    )
    mc = multidms.model_collection.ModelCollection(fit_models_df)
    chart, data = mc.mut_param_dataset_correlation(return_data=True)

    assert np.all(data["correlation"] == 1.0)


def test_ModelCollection_charts():
    """
    Test fitting two different models in
    parallel using multidms.model_collection.fit_models
    """
    data = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=False,
        include_counts=False,
    )
    params = {
        "dataset": [data],
        "maxiter": [2],
        "scale_coeff_lasso_shift": [0.0, 1e-5],
    }
    _, _, fit_models_df = multidms.model_collection.fit_models(
        params,
        n_threads=-1,
    )
    mc = multidms.model_collection.ModelCollection(fit_models_df)

    mc.mut_param_heatmap(query="scale_coeff_lasso_shift == 0.0")
    mc.shift_sparsity()


def test_ModelCollection_get_conditional_loss_df():
    """
    Test that correctness of the conditional loss df
    format and values by comparing the results of
    ModelCollection.get_conditional_loss_df to the
    results of Model.conditional_loss.
    """
    params = {
        "dataset": [data],
        "maxiter": [2],
        "scale_coeff_lasso_shift": [0.0, 1e-5],
    }
    _, _, fit_models_df = multidms.model_collection.fit_models(
        params,
        n_threads=-1,
    )
    mc = multidms.model_collection.ModelCollection(fit_models_df)
    df_loss = mc.get_conditional_loss_df()
    # without validation loss, we expect the loss dataframe
    # to have a row for each model-condition pair + total loss
    n_expected_training_loss_rows = len(mc.fit_models) * (len(data.conditions) + 1)
    assert df_loss.shape[0] == n_expected_training_loss_rows

    mc.add_validation_loss(TEST_FUNC_SCORES)
    df_loss = mc.get_conditional_loss_df()
    # with validation loss, we expect the loss dataframe
    # to have a row for each model-condition-split (training/validation) pair
    # + total loss
    assert df_loss.shape[0] == n_expected_training_loss_rows * 2
