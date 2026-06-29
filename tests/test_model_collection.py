"""Tests for the ModelCollection class and related fitting functions."""

# ruff: noqa: D102

import pytest
import pandas as pd
import numpy as np
from io import StringIO

import multidms
from multidms.model_collection import (
    ModelCollection,
    ModelCollectionFitError,
    _assert_no_nan,
    _extract_seed,
    concat_path_trajectories,
    fit_models,
    fit_models_path,
    fit_one_model,
    stack_fit_models,
)

# Nearly every test here fits a JAX model (directly or via fixtures) — slow on
# CI. Deselected by default (pyproject `addopts = "-m 'not slow'"`); run on
# push-to-main / release.
pytestmark = pytest.mark.slow

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

    def test_plot_convergence_trajectory(self, collection):
        import altair as alt

        chart = collection.plot_convergence_trajectory()
        assert isinstance(chart, (alt.Chart, alt.LayerChart))
        # Verify chart produces a valid Vega-Lite spec
        chart.to_dict()

    def test_plot_convergence_trajectory_standalone(self, collection):
        """Test the standalone plot.convergence_trajectory function."""
        import altair as alt
        import multidms.plot

        df = collection.convergence_trajectory_df()
        chart = multidms.plot.convergence_trajectory(
            df, id_cols=["dataset_name", "fusionreg"]
        )
        assert isinstance(chart, (alt.Chart, alt.LayerChart))
        chart.to_dict()

    def test_plot_convergence_trajectory_no_id_cols(self, collection):
        """Test convergence plot with no id_cols."""
        import altair as alt
        import multidms.plot

        # Get trajectory from a single model
        model = collection.fit_models.iloc[0].model
        df = model.convergence_trajectory_df
        chart = multidms.plot.convergence_trajectory(df)
        assert isinstance(chart, (alt.Chart, alt.LayerChart))
        chart.to_dict()

    def test_plot_convergence_trajectory_custom_groups(self, collection):
        """Test convergence plot with custom trajectory groups."""
        import altair as alt
        import multidms.plot

        df = collection.convergence_trajectory_df()
        custom_groups = {
            "loss": ["loss_trajectory", "loss_per_variant_trajectory"],
        }
        chart = multidms.plot.convergence_trajectory(
            df,
            trajectory_groups=custom_groups,
            init_group="loss",
        )
        assert isinstance(chart, (alt.Chart, alt.LayerChart))
        chart.to_dict()

    def test_per_condition_loss_in_convergence_trajectory(self, collection):
        """Test that per-condition loss columns appear in trajectory DataFrame."""
        df = collection.convergence_trajectory_df()
        # Should have per-condition loss columns for each condition
        loss_cols = [
            c
            for c in df.columns
            if c.startswith("loss_")
            and c not in {"loss_trajectory", "loss_per_variant_trajectory"}
            and not c.startswith("loss_per_variant_")
        ]
        assert len(loss_cols) > 0

        loss_pv_cols = [
            c
            for c in df.columns
            if c.startswith("loss_per_variant_") and c != "loss_per_variant_trajectory"
        ]
        assert len(loss_pv_cols) > 0


# ========== _detect_per_condition_groups tests ==========


class TestDetectPerConditionGroups:
    """Tests for _detect_per_condition_groups with loss columns."""

    def test_detects_loss_per_condition(self):
        """Test that loss_{cond} columns are grouped as loss_per_condition."""
        from multidms.plot import _detect_per_condition_groups

        columns = [
            "loss_trajectory",
            "loss_per_variant_trajectory",
            "loss_Delta",
            "loss_Omicron",
            "loss_per_variant_Delta",
            "loss_per_variant_Omicron",
            "beta0_Delta",
            "beta0_Omicron",
            "sparsity_Omicron",
        ]
        groups = _detect_per_condition_groups(columns)
        assert "loss_per_condition" in groups
        assert sorted(groups["loss_per_condition"]) == ["loss_Delta", "loss_Omicron"]

    def test_detects_loss_per_variant_per_condition(self):
        """Test that loss_per_variant_{cond} columns are grouped correctly."""
        from multidms.plot import _detect_per_condition_groups

        columns = [
            "loss_trajectory",
            "loss_per_variant_trajectory",
            "loss_Delta",
            "loss_Omicron",
            "loss_per_variant_Delta",
            "loss_per_variant_Omicron",
            "beta0_Delta",
        ]
        groups = _detect_per_condition_groups(columns)
        assert "loss_per_variant_per_condition" in groups
        assert sorted(groups["loss_per_variant_per_condition"]) == [
            "loss_per_variant_Delta",
            "loss_per_variant_Omicron",
        ]

    def test_no_column_in_multiple_groups(self):
        """Test that no column appears in more than one group."""
        from multidms.plot import _detect_per_condition_groups

        columns = [
            "loss_trajectory",
            "loss_per_variant_trajectory",
            "loss_Delta",
            "loss_Omicron",
            "loss_per_variant_Delta",
            "loss_per_variant_Omicron",
            "alpha_Delta",
            "alpha_Omicron",
            "beta0_Delta",
            "beta0_Omicron",
            "sparsity_Omicron",
        ]
        groups = _detect_per_condition_groups(columns)
        all_cols = []
        for cols in groups.values():
            all_cols.extend(cols)
        assert len(all_cols) == len(
            set(all_cols)
        ), f"Duplicate columns across groups: {all_cols}"

    def test_base_cols_excluded(self):
        """Test that static base columns are not matched by prefix detection."""
        from multidms.plot import _detect_per_condition_groups

        columns = [
            "loss_trajectory",
            "loss_per_variant_trajectory",
            "loss_Delta",
        ]
        groups = _detect_per_condition_groups(columns)
        if "loss_per_condition" in groups:
            assert "loss_trajectory" not in groups["loss_per_condition"]
        if "loss_per_variant_per_condition" in groups:
            assert (
                "loss_per_variant_trajectory"
                not in groups["loss_per_variant_per_condition"]
            )


# ========== fit_models_path tests ==========


_PATH_FIT_KWARGS = dict(
    ge_type="Identity",
    l2reg=0.0,
    beta0_ridge=0.0,
    maxiter=3,
    tol=1e-4,
    warmstart=False,
)


class TestFitModelsPath:
    """Tests for fit_models_path() and concat_path_trajectories()."""

    @staticmethod
    def _fusionreg_path():
        return [0.0, 1e-5, 1e-4]

    def _path_params(self, datasets, **overrides):
        params = {
            "dataset": datasets,
            "fusionreg": list(self._fusionreg_path()),
            **{k: [v] for k, v in _PATH_FIT_KWARGS.items()},
        }
        params.update(overrides)
        return params

    def test_schema_parity(self, replicate_data):
        """fit_models_path returns the same column set as fit_models."""
        rep1, rep2 = replicate_data
        path_params = self._path_params([rep1, rep2])
        _, _, path_df = fit_models_path(path_params)
        _, _, indep_df = fit_models(path_params, failures="tolerate")
        assert set(path_df.columns) == set(indep_df.columns)
        expected_rows = 2 * len(self._fusionreg_path())  # datasets × path
        assert len(path_df) == expected_rows

    def test_extract_seed_round_trip(self, simple_data):
        """_extract_seed returns exactly the fitted (β, β0, α) of the model."""
        import jax.numpy as jnp

        # Fit a single model to produce a realistic source.
        params = {
            "dataset": [simple_data],
            "fusionreg": [0.0],
            **{k: [v] for k, v in _PATH_FIT_KWARGS.items()},
        }
        _, _, df = fit_models_path(params)
        model = df.iloc[0]["model"]

        beta_init, beta0_init, alpha_init = _extract_seed(model)
        for d in model.data.conditions:
            assert jnp.array_equal(beta_init[d], model.params.φ[d].β)
            assert jnp.array_equal(
                jnp.asarray(beta0_init[d]),
                jnp.asarray(model.params.φ[d].β0),
            )
        # share_alpha=True ⇒ α is a scalar (not a dict)
        assert not isinstance(alpha_init, dict)
        assert jnp.array_equal(
            jnp.asarray(alpha_init),
            jnp.asarray(model.params.α),
        )

    def test_path_step_rows_have_no_jax_seed_leakage(self, simple_data):
        """Path DF is schema-clean: beta/beta0/alpha_init cells are not jax or dict.

        The row returned for path steps k>0 is seeded from the previous
        model, but the seed *values* must not live in the row — they'd be
        jax arrays and dicts-of-jax, which break pandas .apply(str),
        groupby, and pickling round-trips that callers do on fit_models()
        output. Step-0 rows inherit whatever the user passed (usually
        None) so they're clean by construction; only steps k>0 are at
        risk.
        """
        params = self._path_params([simple_data])
        _, _, df = fit_models_path(params)
        for col in ("beta_init", "beta0_init", "alpha_init"):
            assert col in df.columns
            for val in df[col]:
                assert val is None, (
                    f"path DF col '{col}' holds non-None value of type "
                    f"{type(val).__name__}: {val!r}"
                )

    def test_schema_matches_fit_models_exactly(self, simple_data):
        """fit_models and fit_models_path produce identical-shape DataFrames.

        "Identical shape" means same columns AND no column of the path
        DataFrame holds a dict or a jax.Array — the same contract as
        fit_models output, so a ModelCollection built from either is
        indistinguishable.
        """
        import jax

        params = {
            "dataset": [simple_data],
            "fusionreg": [0.0, 1e-5, 1e-4],
            **{k: [v] for k, v in _PATH_FIT_KWARGS.items()},
        }
        _, _, path_df = fit_models_path(params)
        _, _, indep_df = fit_models(params, failures="tolerate")
        assert set(path_df.columns) == set(indep_df.columns)
        for col in path_df.columns:
            if col == "model":
                continue
            for val in path_df[col]:
                assert not isinstance(val, (dict, jax.Array)), (
                    f"path DF col '{col}' leaked a {type(val).__name__}: " f"{val!r}"
                )

    def test_constant_fusionreg_identity(self, simple_data):
        """Constant-fusionreg path: repeated steps converge to the same point.

        Run each step to a tight tolerance so the first step has already
        converged by the time step 2 inherits it — further iterations
        should barely move the parameters.
        """
        import jax.numpy as jnp

        params = {
            "dataset": [simple_data],
            "fusionreg": [0.0, 0.0, 0.0],
            "ge_type": ["Identity"],
            "l2reg": [0.0],
            "beta0_ridge": [0.0],
            "maxiter": [200],
            "tol": [1e-10],
            "warmstart": [False],
        }
        _, _, df = fit_models_path(params)
        assert len(df) == 3
        ref = df.iloc[0]["model"]
        for i in range(1, len(df)):
            cur = df.iloc[i]["model"]
            for d in ref.data.conditions:
                assert jnp.allclose(ref.params.φ[d].β, cur.params.φ[d].β, atol=1e-5)
                assert jnp.allclose(
                    jnp.asarray(ref.params.φ[d].β0),
                    jnp.asarray(cur.params.φ[d].β0),
                    atol=1e-5,
                )
            assert jnp.allclose(
                jnp.asarray(ref.params.α),
                jnp.asarray(cur.params.α),
                atol=1e-5,
            )

    def test_single_step_matches_fit_models(self, simple_data):
        """Single-step path reproduces fit_models' single-point output."""
        import jax.numpy as jnp

        params = {
            "dataset": [simple_data],
            "fusionreg": [0.0],
            **{k: [v] for k, v in _PATH_FIT_KWARGS.items()},
        }
        _, _, path_df = fit_models_path(params)
        _, _, indep_df = fit_models(params, failures="tolerate")
        assert len(path_df) == len(indep_df) == 1
        m_path = path_df.iloc[0]["model"]
        m_indep = indep_df.iloc[0]["model"]
        for d in m_path.data.conditions:
            assert jnp.allclose(m_path.params.φ[d].β, m_indep.params.φ[d].β, atol=1e-6)

    def test_order_invariance(self, replicate_data):
        """Shuffling dataset order does not change per-dataset fits.

        Uses a short 2-step path and asserts no steps failed so that
        memory pressure (CI runners compile+hold a JAX kernel per step)
        surfaces as a clear "n_failed > 0" failure rather than as a
        misleading length mismatch on the merged DataFrame. Clears
        JAX compilation caches between the two path fits because on
        memory-constrained runners (~7GB) the per-step kernels can
        accumulate enough to hit ``LLVM compilation error: Cannot
        allocate memory`` on the second call.
        """
        import jax
        import jax.numpy as jnp

        rep1, rep2 = replicate_data
        short_path = {"fusionreg": [0.0, 1e-5]}
        n_ab, f_ab, df_ab = fit_models_path(
            self._path_params([rep1, rep2], **short_path)
        )
        jax.clear_caches()
        n_ba, f_ba, df_ba = fit_models_path(
            self._path_params([rep2, rep1], **short_path)
        )
        assert f_ab == 0, f"forward path had {f_ab} step failure(s)"
        assert f_ba == 0, f"reverse path had {f_ba} step failure(s)"
        for name in ("rep1", "rep2"):
            sub_ab = df_ab[df_ab["dataset_name"] == name].sort_values("fusionreg")
            sub_ba = df_ba[df_ba["dataset_name"] == name].sort_values("fusionreg")
            assert len(sub_ab) == len(sub_ba) == len(short_path["fusionreg"])
            for row_ab, row_ba in zip(sub_ab.itertuples(), sub_ba.itertuples()):
                m_ab = row_ab.model
                m_ba = row_ba.model
                for d in m_ab.data.conditions:
                    assert jnp.allclose(
                        m_ab.params.φ[d].β, m_ba.params.φ[d].β, atol=1e-6
                    )

    def test_monotone_sparsity(self, simple_data):
        """Sparsity is non-decreasing along the fusionreg path (per condition)."""
        params = self._path_params(
            [simple_data],
            fusionreg=[0.0, 1e-4, 1e-3, 1e-2],
        )
        _, _, df = fit_models_path(params)
        df = df.sort_values("fusionreg").reset_index(drop=True)
        non_ref = [c for c in df.iloc[0]["model"].data.conditions if c != "a"]
        slack = 1e-8
        for cond in non_ref:
            sparsities = [
                float(
                    row["model"].convergence_trajectory_df[f"sparsity_{cond}"].iloc[-1]
                )
                for _, row in df.iterrows()
            ]
            for i in range(1, len(sparsities)):
                assert sparsities[i] + slack >= sparsities[i - 1], (
                    f"sparsity dropped at fusionreg={df.loc[i, 'fusionreg']} "
                    f"for condition {cond}: {sparsities}"
                )

    def test_nan_guard_raises(self):
        """_assert_no_nan raises on NaN in β, β0, or α."""
        import jax.numpy as jnp

        class _FakeLatent:
            def __init__(self, β, β0):
                self.β = β
                self.β0 = β0

        class _FakeParams:
            def __init__(self, φ, α):
                self.φ = φ
                self.α = α

        class _FakeData:
            def __init__(self, conds):
                self.conditions = conds

        class _FakeModel:
            def __init__(self, φ, α, conds):
                self.params = _FakeParams(φ, α)
                self.data = _FakeData(conds)

        good = _FakeLatent(jnp.array([0.0, 1.0]), jnp.array(0.0))
        # β NaN
        bad_beta = _FakeLatent(jnp.array([jnp.nan, 1.0]), jnp.array(0.0))
        with pytest.raises(ModelCollectionFitError, match="NaN in β"):
            _assert_no_nan(
                _FakeModel({"a": bad_beta, "b": good}, jnp.array(1.0), ["a", "b"])
            )
        # β0 NaN
        bad_b0 = _FakeLatent(jnp.array([0.0]), jnp.array(jnp.nan))
        with pytest.raises(ModelCollectionFitError, match="NaN in β0"):
            _assert_no_nan(_FakeModel({"a": bad_b0}, jnp.array(1.0), ["a"]))
        # α NaN (scalar)
        with pytest.raises(ModelCollectionFitError, match="NaN in α"):
            _assert_no_nan(_FakeModel({"a": good}, jnp.array(jnp.nan), ["a"]))
        # α NaN (dict)
        with pytest.raises(ModelCollectionFitError, match="NaN in α"):
            _assert_no_nan(
                _FakeModel(
                    {"a": good},
                    {"a": jnp.array(jnp.nan)},
                    ["a"],
                )
            )

    def test_nonzero_first_step_warns(self, simple_data):
        """Starting the path at fusionreg > 0 emits a warning."""
        params = {
            "dataset": [simple_data],
            "fusionreg": [1e-5, 1e-4],
            **{k: [v] for k, v in _PATH_FIT_KWARGS.items()},
        }
        with pytest.warns(UserWarning, match="unregularized"):
            fit_models_path(params)

    def test_verbose_in_params_does_not_collide(self, simple_data):
        """User-supplied verbose in params must not double-pass.

        fit_models_path takes its own verbose= kwarg for convenience, and
        fit_one_model also accepts verbose — if a user puts verbose into
        params, a naïve implementation would pass it twice and raise
        TypeError. The driver uses setdefault so the user value wins.
        """
        params = {
            "dataset": [simple_data],
            "fusionreg": [0.0, 1e-5],
            "verbose": [False],  # user sets it in params
            **{k: [v] for k, v in _PATH_FIT_KWARGS.items()},
        }
        n_fit, n_failed, df = fit_models_path(params)
        assert n_fit == 2 and n_failed == 0


class TestConcatPathTrajectories:
    """Tests for concat_path_trajectories()."""

    def test_concat_structure(self, simple_data):
        path_params = {
            "dataset": [simple_data],
            "fusionreg": [0.0, 1e-5],
            **{k: [v] for k, v in _PATH_FIT_KWARGS.items()},
        }
        _, _, df = fit_models_path(path_params)
        long = concat_path_trajectories(df)
        # Non-empty, and contains expected columns
        assert len(long) > 0
        for col in (
            "path_id",
            "step_index",
            "fusionreg",
            "iteration_within_step",
            "iteration_global",
        ):
            assert col in long.columns
        # iteration_global is strictly monotone within a path
        for _, sub in long.groupby("path_id"):
            glob = sub["iteration_global"].to_numpy()
            assert all(glob[i] < glob[i + 1] for i in range(len(glob) - 1))
        # fusionreg is constant within each step_index
        for (pid, step), sub in long.groupby(["path_id", "step_index"]):
            assert sub["fusionreg"].nunique() == 1
        # Row count matches sum of per-step trajectories
        expected = sum(
            len(row["model"].convergence_trajectory_df) for _, row in df.iterrows()
        )
        assert len(long) == expected

    def test_empty_input_returns_empty_df(self):
        empty = pd.DataFrame(columns=["dataset_name", "fusionreg", "model"])
        out = concat_path_trajectories(empty)
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0
