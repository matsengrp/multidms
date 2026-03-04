"""Tests for the ModelCollection class and related fitting functions."""

# ruff: noqa: D102

import pytest
import pandas as pd
import numpy as np
from io import StringIO

import multidms
from multidms.model_collection import (
    fit_one_model,
    stack_fit_models,
    fit_models,
    ModelCollection,
)

# ========== Test Data ==========

TEST_FUNC_SCORES = pd.read_csv(
    StringIO(
        """
condition,aa_substitutions,func_score
a,,0.0
a,M1E,2.0
a,G3R,-7.0
a,G3P,-0.5
a,M1W,2.3
b,,0.0
b,M1E,1.0
b,P3R,-5.0
b,P3G,0.4
b,M1E P3G,2.7
b,M1E P3R,-2.7
b,P2T,0.3
        """
    )
).fillna({"aa_substitutions": ""})

# Validation data excludes P2T (unseen mutation for reference condition "a")
TEST_VALIDATION_SCORES = TEST_FUNC_SCORES[
    ~TEST_FUNC_SCORES["aa_substitutions"].str.contains("P2T", na=False)
].reset_index(drop=True)


# ========== Fixtures ==========


@pytest.fixture(scope="session")
def simple_data():
    """A single Data object for basic fitting tests."""
    return multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=True,
        name="test_data",
        include_counts=False,
    )


@pytest.fixture(scope="session")
def replicate_data():
    """Two Data objects from identical data with different names."""
    rep1 = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=True,
        name="rep1",
        include_counts=False,
    )
    rep2 = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=True,
        name="rep2",
        include_counts=False,
    )
    return rep1, rep2


@pytest.fixture(scope="module")
def fit_models_df():
    """DataFrame from fitting 2 models with different fusionreg values."""
    data = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=True,
        name="test_data",
        include_counts=False,
    )
    series_list = []
    for fusionreg in [0.0, 1e-5]:
        s = fit_one_model(
            dataset=data,
            fusionreg=fusionreg,
            maxiter=2,
            warmstart=False,
            verbose=False,
        )
        series_list.append(s)
    return stack_fit_models(series_list)


@pytest.fixture(scope="module")
def replicate_fit_models_df():
    """DataFrame from fitting 2 models on 2 replicate datasets."""
    rep1 = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=True,
        name="rep1",
        include_counts=False,
    )
    rep2 = multidms.Data(
        TEST_FUNC_SCORES,
        alphabet=multidms.AAS_WITHSTOP,
        reference="a",
        assert_site_integrity=True,
        name="rep2",
        include_counts=False,
    )
    series_list = []
    for dataset in [rep1, rep2]:
        for fusionreg in [0.0, 1e-5]:
            s = fit_one_model(
                dataset=dataset,
                fusionreg=fusionreg,
                maxiter=2,
                warmstart=False,
                verbose=False,
            )
            series_list.append(s)
    return stack_fit_models(series_list)


@pytest.fixture(scope="module")
def collection(fit_models_df):
    """ModelCollection from the fit_models_df fixture."""
    return ModelCollection(fit_models_df.copy())


@pytest.fixture(scope="module")
def replicate_collection(replicate_fit_models_df):
    """ModelCollection from the replicate_fit_models_df fixture."""
    return ModelCollection(replicate_fit_models_df.copy())


# ========== fit_one_model tests ==========


class TestFitOneModel:
    """Tests for fit_one_model()."""

    def test_returns_series(self, simple_data):
        result = fit_one_model(
            dataset=simple_data,
            maxiter=1,
            warmstart=False,
            verbose=False,
        )
        assert isinstance(result, pd.Series)

    def test_contains_model(self, simple_data):
        result = fit_one_model(
            dataset=simple_data,
            maxiter=1,
            warmstart=False,
            verbose=False,
        )
        assert isinstance(result["model"], multidms.Model)

    def test_records_dataset_name(self, simple_data):
        result = fit_one_model(
            dataset=simple_data,
            maxiter=1,
            warmstart=False,
            verbose=False,
        )
        assert result["dataset_name"] == "test_data"

    def test_records_fit_time(self, simple_data):
        result = fit_one_model(
            dataset=simple_data,
            maxiter=1,
            warmstart=False,
            verbose=False,
        )
        assert "fit_time" in result.index
        assert isinstance(result["fit_time"], (int, float))

    def test_hyperparameters_stored(self, simple_data):
        result = fit_one_model(
            dataset=simple_data,
            ge_type="Identity",
            l2reg=0.1,
            fusionreg=0.5,
            warmstart=False,
            maxiter=1,
            verbose=False,
        )
        assert result["ge_type"] == "Identity"
        assert result["l2reg"] == 0.1
        assert result["fusionreg"] == 0.5

    def test_expected_keys(self, simple_data):
        result = fit_one_model(
            dataset=simple_data,
            maxiter=1,
            warmstart=False,
            verbose=False,
        )
        expected_keys = {
            "model",
            "dataset_name",
            "fit_time",
            "ge_type",
            "l2reg",
            "fusionreg",
            "beta0_ridge",
            "loss_type",
            "maxiter",
            "tol",
            "warmstart",
        }
        assert expected_keys.issubset(set(result.index))


# ========== stack_fit_models tests ==========


class TestStackFitModels:
    """Tests for stack_fit_models()."""

    def test_stacks_series_into_df(self, simple_data):
        s1 = fit_one_model(
            dataset=simple_data,
            maxiter=1,
            warmstart=False,
            verbose=False,
        )
        s2 = fit_one_model(
            dataset=simple_data,
            fusionreg=0.01,
            maxiter=1,
            warmstart=False,
            verbose=False,
        )
        df = stack_fit_models([s1, s2])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_preserves_columns(self, simple_data):
        s1 = fit_one_model(
            dataset=simple_data,
            maxiter=1,
            warmstart=False,
            verbose=False,
        )
        s2 = fit_one_model(
            dataset=simple_data,
            fusionreg=0.01,
            maxiter=1,
            warmstart=False,
            verbose=False,
        )
        df = stack_fit_models([s1, s2])
        assert set(s1.index) == set(df.columns)


# ========== fit_models tests ==========


class TestFitModels:
    """Tests for fit_models()."""

    def test_basic_fitting(self, simple_data):
        params = {
            "dataset": [simple_data],
            "maxiter": [1],
            "warmstart": [False],
            "fusionreg": [0.0],
        }
        n_fit, n_failed, df = fit_models(params)
        assert n_fit == 1
        assert n_failed == 0
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_parameter_explosion(self, simple_data):
        params = {
            "dataset": [simple_data],
            "maxiter": [1],
            "warmstart": [False],
            "fusionreg": [0.0, 1e-5],
        }
        n_fit, n_failed, df = fit_models(params)
        assert n_fit == 2
        assert len(df) == 2

    def test_n_processes_explicit(self, simple_data):
        params = {
            "dataset": [simple_data],
            "maxiter": [1],
            "warmstart": [False],
            "fusionreg": [0.0],
        }
        n_fit, n_failed, df = fit_models(params, n_processes=1)
        assert n_fit == 1
        assert isinstance(df, pd.DataFrame)

    def test_failures_tolerate(self, simple_data):
        params = {
            "dataset": [simple_data],
            "maxiter": [1],
            "warmstart": [False],
            "fusionreg": [0.0],
        }
        n_fit, n_failed, df = fit_models(params, failures="tolerate")
        assert n_fit >= 1

    def test_invalid_failures_raises(self, simple_data):
        params = {
            "dataset": [simple_data],
            "maxiter": [1],
            "warmstart": [False],
            "fusionreg": [0.0],
        }
        with pytest.raises(ValueError):
            fit_models(params, failures="bad_value")

    def test_gpu_ids_and_n_processes_mutually_exclusive(self, simple_data):
        params = {
            "dataset": [simple_data],
            "maxiter": [1],
            "warmstart": [False],
            "fusionreg": [0.0],
        }
        with pytest.raises(ValueError, match="Cannot specify both"):
            fit_models(params, gpu_ids=[0], n_processes=2)

    def test_empty_gpu_ids_raises(self, simple_data):
        params = {
            "dataset": [simple_data],
            "maxiter": [1],
            "warmstart": [False],
            "fusionreg": [0.0],
        }
        with pytest.raises(ValueError, match="non-empty"):
            fit_models(params, gpu_ids=[])

    def test_invalid_n_processes_raises(self, simple_data):
        params = {
            "dataset": [simple_data],
            "maxiter": [1],
            "warmstart": [False],
            "fusionreg": [0.0],
        }
        with pytest.raises(ValueError, match="n_processes must be >= 1"):
            fit_models(params, n_processes=0)

    def test_n_threads_deprecation_warning(self, simple_data):
        params = {
            "dataset": [simple_data],
            "maxiter": [1],
            "warmstart": [False],
            "fusionreg": [0.0],
        }
        with pytest.warns(DeprecationWarning, match="n_threads is deprecated"):
            n_fit, n_failed, df = fit_models(params, n_threads=1)
        assert n_fit == 1


# ========== ModelCollection.__init__ tests ==========


class TestModelCollectionInit:
    """Tests for ModelCollection initialization and properties."""

    def test_conditions(self, collection, simple_data):
        assert collection.conditions == simple_data.conditions

    def test_reference(self, collection, simple_data):
        assert collection.reference == simple_data.reference

    def test_shared_mutations(self, collection, simple_data):
        assert set(collection.shared_mutations) == set(simple_data.mutations)

    def test_all_mutations(self, collection, simple_data):
        assert set(collection.all_mutations) == set(simple_data.mutations)

    def test_site_map_union(self, collection, simple_data):
        assert collection.site_map_union.equals(simple_data.site_map)

    def test_convergence_column(self, collection):
        assert "converged" in collection.fit_models.columns
        assert collection.fit_models["converged"].dtype == bool

    def test_training_loss_columns(self, collection):
        for condition in collection.conditions:
            assert f"{condition}_loss_training" in collection.fit_models.columns
        assert "total_loss_training" in collection.fit_models.columns

    def test_condition_colors(self, collection):
        assert isinstance(collection.condition_colors, dict)
        assert set(collection.conditions).issubset(
            set(collection.condition_colors.keys())
        )

    def test_mismatched_references_raises(self, simple_data):
        data_diff_ref = multidms.Data(
            TEST_FUNC_SCORES,
            alphabet=multidms.AAS_WITHSTOP,
            reference="b",
            assert_site_integrity=False,
            name="diff_ref",
            include_counts=False,
        )
        s1 = fit_one_model(
            dataset=simple_data, maxiter=1, warmstart=False, verbose=False
        )
        s2 = fit_one_model(
            dataset=data_diff_ref, maxiter=1, warmstart=False, verbose=False
        )
        df = stack_fit_models([s1, s2])
        with pytest.raises(ValueError, match="reference"):
            ModelCollection(df)

    def test_mismatched_conditions_raises(self):
        df_a_only = pd.read_csv(
            StringIO(
                """
condition,aa_substitutions,func_score
a,,0.0
a,M1E,2.0
a,G3R,-7.0
"""
            )
        )
        df_c_only = pd.read_csv(
            StringIO(
                """
condition,aa_substitutions,func_score
a,,0.0
a,M1E,2.0
c,,0.0
c,M1E,1.5
"""
            )
        )
        data1 = multidms.Data(
            df_a_only,
            reference="a",
            name="data1",
            include_counts=False,
        )
        data2 = multidms.Data(
            df_c_only,
            reference="a",
            name="data2",
            include_counts=False,
        )
        s1 = fit_one_model(dataset=data1, maxiter=1, warmstart=False, verbose=False)
        s2 = fit_one_model(dataset=data2, maxiter=1, warmstart=False, verbose=False)
        df = stack_fit_models([s1, s2])
        with pytest.raises(ValueError, match="conditions"):
            ModelCollection(df)


# ========== split_apply_combine_muts tests ==========


class TestSplitApplyCombineMuts:
    """Tests for ModelCollection.split_apply_combine_muts()."""

    def test_tuple_groupby(self, collection):
        result = collection.split_apply_combine_muts(
            groupby=("dataset_name", "fusionreg")
        )
        assert isinstance(result, pd.DataFrame)
        assert list(result.index.names) == ["dataset_name", "fusionreg"]
        assert "mutation" in result.columns

    def test_string_groupby(self, collection):
        result = collection.split_apply_combine_muts(groupby="fusionreg")
        assert isinstance(result, pd.DataFrame)
        assert list(result.index.names) == ["fusionreg"]

    def test_none_groupby(self, collection):
        result = collection.split_apply_combine_muts(groupby=None)
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == "mutation"

    def test_query_filtering(self, collection):
        result = collection.split_apply_combine_muts(
            groupby=None, query="fusionreg == 0.0"
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_inner_merge_false(self, replicate_collection):
        result_inner = replicate_collection.split_apply_combine_muts(
            groupby=None, inner_merge_dataset_muts=True
        )
        result_outer = replicate_collection.split_apply_combine_muts(
            groupby=None, inner_merge_dataset_muts=False
        )
        assert len(result_outer) >= len(result_inner)

    def test_invalid_groupby_raises(self, collection):
        with pytest.raises(ValueError):
            collection.split_apply_combine_muts(groupby=42)

    def test_empty_query_raises(self, collection):
        with pytest.raises(ValueError, match="invalid query"):
            collection.split_apply_combine_muts(
                groupby=None, query="fusionreg == 999999"
            )

    def test_string_columns_excluded(self, collection):
        """Verify string columns (wts, muts) are excluded from aggregation."""
        result = collection.split_apply_combine_muts(groupby=None)
        for col in ["wts", "muts"]:
            assert col not in result.columns


# ========== add_validation_loss tests ==========


class TestAddEvalLoss:
    """Tests for ModelCollection.add_eval_loss()."""

    def test_adds_validation_columns(self, fit_models_df):
        mc = ModelCollection(fit_models_df.copy())
        mc.add_eval_loss(TEST_VALIDATION_SCORES)
        for condition in mc.conditions:
            assert f"{condition}_loss_validation" in mc.fit_models.columns

    def test_total_loss_validation(self, fit_models_df):
        mc = ModelCollection(fit_models_df.copy())
        mc.add_eval_loss(TEST_VALIDATION_SCORES)
        assert "total_loss_validation" in mc.fit_models.columns
        assert mc.fit_models["total_loss_validation"].notna().all()

    def test_dict_input(self, fit_models_df):
        mc = ModelCollection(fit_models_df.copy())
        test_dict = {"test_data": TEST_VALIDATION_SCORES}
        mc.add_eval_loss(test_dict)
        assert "total_loss_validation" in mc.fit_models.columns

    def test_overwrite_false_raises(self, fit_models_df):
        mc = ModelCollection(fit_models_df.copy())
        mc.add_eval_loss(TEST_VALIDATION_SCORES)
        with pytest.raises(ValueError, match="overwrite"):
            mc.add_eval_loss(TEST_VALIDATION_SCORES, overwrite=False)

    def test_overwrite_true_works(self, fit_models_df):
        mc = ModelCollection(fit_models_df.copy())
        mc.add_eval_loss(TEST_VALIDATION_SCORES)
        mc.add_eval_loss(TEST_VALIDATION_SCORES, overwrite=True)
        assert mc.fit_models["total_loss_validation"].notna().all()


# ========== get_conditional_loss_df tests ==========


class TestLossDf:
    """Tests for ModelCollection.loss_df()."""

    def test_returns_correct_columns(self, collection):
        df = collection.loss_df()
        assert "dataset_name" in df.columns
        assert "fusionreg" in df.columns
        assert "condition" in df.columns
        assert "loss" in df.columns
        assert "split" in df.columns

    def test_training_only_rows(self, collection):
        df = collection.loss_df()
        assert set(df["split"].unique()) == {"training"}

    def test_with_validation_loss(self, fit_models_df):
        mc = ModelCollection(fit_models_df.copy())
        mc.add_eval_loss(TEST_VALIDATION_SCORES)
        df = mc.loss_df()
        assert "validation" in df["split"].values

    def test_query_filtering(self, collection):
        df = collection.loss_df(query="fusionreg == 0.0")
        assert all(df["fusionreg"] == 0.0)


# ========== convergence_trajectory_df tests ==========


class TestConvergenceTrajectoryDf:
    """Tests for ModelCollection.convergence_trajectory_df()."""

    def test_combines_trajectories(self, collection):
        df = collection.convergence_trajectory_df()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_id_vars(self, collection):
        df = collection.convergence_trajectory_df()
        assert "dataset_name" in df.columns
        assert "fusionreg" in df.columns

    def test_query_filtering(self, collection):
        df = collection.convergence_trajectory_df(query="fusionreg == 0.0")
        assert all(df["fusionreg"] == 0.0)


# ========== Visualization smoke tests ==========


class TestVisualization:
    """Smoke tests for visualization methods (verify they return charts)."""

    def test_mut_param_heatmap(self, collection):
        import altair as alt

        chart = collection.mut_param_heatmap(query="fusionreg == 0.0")
        assert isinstance(chart, (alt.Chart, alt.LayerChart, alt.VConcatChart))

    def test_mut_param_traceplot(self, collection):
        import altair as alt

        mutations = list(collection.shared_mutations)[:3]
        chart = collection.mut_param_traceplot(mutations=mutations)
        assert isinstance(chart, (alt.Chart, alt.FacetChart))

    def test_shift_sparsity(self, collection):
        import altair as alt

        chart = collection.shift_sparsity()
        assert isinstance(chart, (alt.Chart, alt.FacetChart))

    def test_shift_sparsity_return_data(self, collection):
        result = collection.shift_sparsity(return_data=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[1], pd.DataFrame)

    def test_mut_param_dataset_correlation(self, replicate_collection):
        import altair as alt

        chart = replicate_collection.mut_param_dataset_correlation()
        assert isinstance(chart, (alt.Chart, alt.FacetChart))

    def test_mut_param_dataset_correlation_return_data_r1(self, replicate_collection):
        chart, data = replicate_collection.mut_param_dataset_correlation(
            return_data=True, r=1
        )
        assert isinstance(data, pd.DataFrame)
        assert "correlation" in data.columns

    def test_mut_param_dataset_correlation_return_data_r2(self, replicate_collection):
        chart, data = replicate_collection.mut_param_dataset_correlation(
            return_data=True, r=2
        )
        assert isinstance(data, pd.DataFrame)
        # For identical data, R^2 should be very close to 1.0
        assert np.allclose(data["correlation"], 1.0, atol=0.01)
