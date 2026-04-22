r"""
=====
model
=====

Defines :class:`Model` objects for global epistasis modeling using JAX.
"""
from __future__ import annotations

from typing import Literal
import pandas as pd
import numpy as np

from multidms.data import Data
import multidms.jaxmodels as jaxmodels


class Model:
    r"""
    Model for global epistasis analysis of DMS experiments.

    This class wraps the jaxmodels backend to provide a user-friendly
    interface for fitting global epistasis models to deep mutational
    scanning data.

    Parameters
    ----------
    data : multidms.Data
        Preprocessed DMS data object containing variants and functional scores.
    loss_type : {'functional_score_loss', 'count_loss'}
        Type of loss function to use. 'functional_score_loss' for standard
        functional score fitting, 'count_loss' for count-based enrichment models.
    ge_type : {'Identity', 'Sigmoid'}
        Global epistasis model type. 'Identity' for no global epistasis (linear),
        'Sigmoid' for sigmoidal transformation.
    l2reg : float
        L2 regularization strength for mutation effects (default: 0.0).
    fusionreg : float
        Fusion regularization strength for shift parameters (default: 0.0).
    beta0_ridge : float
        Ridge penalty on squared β0, summed across all conditions
        (reference included; default: 0.0).

    Example
    -------
    >>> import pandas as pd
    >>> from multidms import Data, Model
    >>> df = pd.DataFrame({
    ...     'condition': ['a', 'a', 'b', 'b'],
    ...     'aa_substitutions': ['', 'M1A', '', 'M1A'],
    ...     'func_score': [0.0, 1.2, 0.1, 1.5]
    ... })
    >>> data = Data(df, reference='a')  # doctest: +ELLIPSIS
    >>> model = Model(data, ge_type='Sigmoid', l2reg=0.01)
    >>> model  # doctest: +ELLIPSIS
    Model(ge_type='Sigmoid', loss_type='functional_score_loss')
    """

    def __init__(
        self,
        data: Data,
        loss_type: Literal[
            "functional_score_loss", "count_loss"
        ] = "functional_score_loss",
        ge_type: Literal["Identity", "Sigmoid"] = "Sigmoid",
        l2reg: float = 0.0,
        fusionreg: float = 0.0,
        beta0_ridge: float = 0.0,
        scale_fusion_by_n: bool = False,
    ):
        """Initialize Model with data and hyperparameters."""
        # Validate inputs
        if loss_type not in ["functional_score_loss", "count_loss"]:
            raise ValueError(
                f"loss_type must be 'functional_score_loss' or 'count_loss', "
                f"got {loss_type}"
            )
        if ge_type not in ["Identity", "Sigmoid"]:
            raise ValueError(f"ge_type must be 'Identity' or 'Sigmoid', got {ge_type}")

        # Store configuration
        self._data = data
        self._loss_type = loss_type
        self._ge_type = ge_type
        self._l2reg = l2reg
        self._fusionreg = fusionreg
        self._beta0_ridge = beta0_ridge
        self._scale_fusion_by_n = scale_fusion_by_n

        # Will be populated by fit()
        self._jax_model = None
        self._jax_data_sets = None
        self._convergence_trajectory_df = None
        self._fit_tol = None
        self._loss_fn = None
        self._loss_kwargs = None

    @property
    def data(self) -> Data:
        """The Data object used for model fitting."""
        return self._data

    @property
    def params(self):
        """Model parameters (available after fit)."""
        if self._jax_model is None:
            return None
        return self._jax_model

    @property
    def converged(self) -> bool:
        """Whether the model fitting converged.

        Convergence is determined by whether the objective error
        (relative change in the objective function) at the last iteration
        was below the tolerance used during fitting.
        """
        if (
            self._convergence_trajectory_df is None
            or len(self._convergence_trajectory_df) == 0
        ):
            return False
        return (
            float(
                self._convergence_trajectory_df["objective_error_trajectory"].iloc[-1]
            )
            < self._fit_tol
        )

    @property
    def convergence_trajectory_df(self) -> pd.DataFrame:
        """
        Convergence trajectory with diagnostics over iterations.

        Returns
        -------
        pd.DataFrame
            One row per iteration with columns for overall objective
            (``iteration``, ``objective_total_trajectory``,
            ``objective_error_trajectory``, ``loss_trajectory``,
            ``loss_per_variant_trajectory``), per-condition loss
            (``loss_{cond}``, ``loss_per_variant_{cond}``),
            block-level diagnostics (e.g. ``calibration_error``,
            ``beta0_stepsize``), shared alpha (``alpha``),
            per-condition parameters (``beta0_{cond}``,
            ``sparsity_{cond}``), and ``theta_{cond}`` when count
            data is present.
        """
        return self._convergence_trajectory_df

    # See issue #178 for optimization of re-fitting already fitted models
    def fit(
        self,
        warmstart: bool = True,
        maxiter: int = 10,
        tol: float = 1e-6,
        beta0_init: dict = None,
        beta_init: dict = None,
        alpha_init: float | dict = None,
        share_alpha: bool = True,
        beta_clip_range: tuple = None,
        ge_kwargs: dict = None,
        cal_kwargs: dict = None,
        loss_kwargs: dict = None,
        verbose: bool = True,
    ):
        """
        Fit the model to data.

        Parameters
        ----------
        warmstart : bool
            Whether to use Ridge regression for parameter initialization (default: True).
        maxiter : int
            Maximum number of optimization iterations (default: 10).
        tol : float
            Convergence tolerance on objective function (default: 1e-6).
        beta0_init : dict, optional
            Initial β0 values per condition.
        beta_init : dict, optional
            Initial β values per condition.
        alpha_init : float or dict, optional
            Initial α scaling value. Float applies to all conditions;
            dict maps condition names to per-condition values.
        share_alpha : bool
            If True (default), optimize a single shared α. If False,
            each condition gets its own α.
        beta_clip_range : tuple, optional
            Tuple of (min, max) values for clipping β parameters during optimization.
            Example: (-10.0, 10.0). If None, no clipping is applied.
        ge_kwargs : dict, optional
            Keyword arguments for global epistasis optimizer (e.g., tol, maxiter, maxls).
        cal_kwargs : dict, optional
            Keyword arguments for calibration (α) optimizer (e.g., tol, maxiter, maxls).
        loss_kwargs : dict, optional
            Keyword arguments for the loss function (e.g., δ for Huber loss).
        verbose : bool
            Whether to print progress information during fitting (default: True).

        Returns
        -------
        self
            Returns self for method chaining.
        """
        # Set default kwargs
        if ge_kwargs is None:
            ge_kwargs = {}
        if cal_kwargs is None:
            cal_kwargs = {}
        if loss_kwargs is None:
            loss_kwargs = {}
        # Convert multidms.Data to jaxmodels.Data for each condition
        self._jax_data_sets = {}
        for condition in self._data.conditions:
            self._jax_data_sets[condition] = jaxmodels.Data.from_multidms(
                self._data, condition
            )

        # Set up global epistasis model
        if self._ge_type == "Identity":
            global_epistasis = jaxmodels.Identity()
        elif self._ge_type == "Sigmoid":
            global_epistasis = jaxmodels.Sigmoid()
        else:
            raise ValueError(f"Unknown ge_type: {self._ge_type}")

        # Set up loss function
        if self._loss_type == "functional_score_loss":
            loss_fn = jaxmodels.functional_score_loss
        elif self._loss_type == "count_loss":
            loss_fn = jaxmodels.count_loss
        else:
            raise ValueError(f"Unknown loss_type: {self._loss_type}")

        # Store fit metadata for post-fit properties/methods
        self._fit_tol = tol
        self._loss_fn = loss_fn
        self._loss_kwargs = loss_kwargs

        # Fit model using jaxmodels
        self._jax_model, self._convergence_trajectory_df = jaxmodels.fit(
            data_sets=self._jax_data_sets,
            reference_condition=self._data.reference,
            l2reg=self._l2reg,
            fusionreg=self._fusionreg,
            beta0_ridge=self._beta0_ridge,
            scale_fusion_by_n=self._scale_fusion_by_n,
            block_iters=maxiter,
            block_tol=tol,
            global_epistasis=global_epistasis,
            loss_fn=loss_fn,
            warmstart=warmstart,
            beta0_init=beta0_init,
            beta_init=beta_init,
            alpha_init=alpha_init,
            share_alpha=share_alpha,
            beta_clip_range=beta_clip_range,
            ge_kwargs=ge_kwargs,
            cal_kwargs=cal_kwargs,
            loss_kwargs=loss_kwargs,
            verbose=verbose,
        )

        return self

    def _get_single_mutation_data(self, condition: str) -> jaxmodels.Data:
        """Build jaxmodels.Data encoding each single mutation as a variant.

        For reference-sequence conditions, X is the identity matrix (each row
        encodes one mutation). For non-reference conditions with different WT
        sequences, each row also includes 1s at bundle mutation indices
        (non-identical sites), matching the encoding used by
        ``_encode_variants`` and ``add_phenotypes_to_df``.

        Uses ``Data.single_mut_encodings`` which handles this distinction.

        Parameters
        ----------
        condition : str
            Condition name.

        Returns
        -------
        jaxmodels.Data
            Data object with X of shape (n_mutations, binarylength).
        """
        X = self._data.single_mut_encodings[condition]
        x_wt = self._jax_data_sets[condition].x_wt
        n_mutations = len(self._data.mutations)
        functional_scores = np.zeros(n_mutations)

        return jaxmodels.Data(
            x_wt=x_wt,
            X=X,
            functional_scores=functional_scores,
        )

    # See issue #179 for removal of deprecated phenotype_as_effect parameter
    def get_mutations_df(
        self,
        phenotype_as_effect: bool = True,
        times_seen_threshold: int = 0,
    ) -> pd.DataFrame:
        """
        Extract mutation-level parameters and predicted functional scores.

        Parameters
        ----------
        phenotype_as_effect : bool
            If True, report mutation effects. If False, report raw latent phenotypes.
        times_seen_threshold : int
            Minimum number of times a mutation must be seen in ALL conditions
            to be included. Default is 0 (no filtering).

        Returns
        -------
        pd.DataFrame
            DataFrame with mutations as rows (index) and columns:

            - beta_{condition} for each condition
            - shift_{condition} for each non-reference condition
            - predicted_func_score_{condition} for each condition

            Shift parameters represent the difference in beta values between each
            condition and the reference condition. Predicted functional scores
            are the model's predictions for each single mutation on its
            condition-specific wild-type background.

        Example
        -------
        For a model with conditions ['a', 'b'] where 'a' is reference,
        the returned columns are: ``beta_a``, ``beta_b``, ``shift_b``,
        ``predicted_func_score_a``, ``predicted_func_score_b``.
        One row per mutation.
        """
        if self._jax_model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # Start with mutations_df from data
        mutations_df = self._data.mutations_df.copy().set_index("mutation")

        # Get reference condition betas
        reference_latent = self._jax_model.φ[self._data.reference]
        reference_betas = reference_latent.β

        # Add beta columns for each condition
        for condition in self._data.conditions:
            latent = self._jax_model.φ[condition]
            mutations_df[f"beta_{condition}"] = latent.β

        # Add shift columns for non-reference conditions
        for condition in self._data.conditions:
            if condition != self._data.reference:
                latent = self._jax_model.φ[condition]
                mutations_df[f"shift_{condition}"] = latent.β - reference_betas

        # Add predicted functional score columns
        for condition in self._data.conditions:
            single_mut_data = self._get_single_mutation_data(condition)
            predictions = self._jax_model.predict_score({condition: single_mut_data})
            mutations_df[f"predicted_func_score_{condition}"] = np.array(
                predictions[condition]
            )

        # Filter by times_seen_threshold
        if times_seen_threshold > 0:
            times_seen_cols = [
                c for c in mutations_df.columns if c.startswith("times_seen_")
            ]
            if times_seen_cols:
                mask = mutations_df[times_seen_cols].min(axis=1) >= times_seen_threshold
                mutations_df = mutations_df[mask]

        return mutations_df

    def get_variants_df(self, phenotype_as_effect: bool = True) -> pd.DataFrame:
        """
        Extract variant-level predictions.

        Parameters
        ----------
        phenotype_as_effect : bool
            If True, report effects. If False, report raw latent phenotypes.

        Returns
        -------
        pd.DataFrame
            Variant-level predictions merged with original data.
            Includes columns:

            - ``predicted_func_score``: model-predicted functional score,
              i.e. ``α * (g(φ(X)) - g(φ(x_wt)))``
            - ``predicted_latent``: latent phenotype ``φ(X)``
            - ``predicted_fitness``: predicted fitness in ``g(φ)`` space,
              i.e. ``predicted_func_score / α + g(φ(x_wt))``
            - ``measured_fitness``: measured fitness in ``g(φ)`` space,
              i.e. ``func_score / α + g(φ(x_wt))``
        """
        if self._jax_model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # Get predictions from jaxmodels
        predictions = self._jax_model.predict_score(self._jax_data_sets)

        # Build DataFrame
        result_rows = []
        for condition in self._data.conditions:
            cond_data = self._data.variants_df[
                self._data.variants_df.condition == condition
            ].copy()

            # jaxmodels predictions exclude WT (index 0), so we need to handle that
            pred_scores = predictions[condition]

            # Add WT prediction (should be 0)
            full_predictions = np.concatenate([[0.0], np.array(pred_scores)])

            # Add predictions to condition data
            cond_data["predicted_func_score"] = full_predictions[: len(cond_data)]

            # Latent phenotype: φ(X) for each variant
            φ = self._jax_model.φ[condition]
            X = self._jax_data_sets[condition].X
            x_wt = self._jax_data_sets[condition].x_wt
            φ_X = np.array(φ(X))
            φ_wt = float(φ(x_wt))
            # WT is index 0, its latent phenotype is φ(x_wt)
            full_latent = np.concatenate([[φ_wt], φ_X])
            cond_data["predicted_latent"] = full_latent[: len(cond_data)]

            # Fitness: back-transform into g(φ) space
            _α = self._jax_model.α
            α = float(_α[condition] if isinstance(_α, dict) else _α)
            g_φ_wt = float(self._jax_model.global_epistasis(φ(x_wt)))

            cond_data["predicted_fitness"] = (
                cond_data["predicted_func_score"] / α + g_φ_wt
            )
            cond_data["measured_fitness"] = cond_data["func_score"] / α + g_φ_wt

            result_rows.append(cond_data)

        return pd.concat(result_rows, ignore_index=True)

    def _encode_variants(
        self, df, condition_col="condition", substitutions_col="aa_substitutions"
    ):
        """Encode a DataFrame of variants into jaxmodels.Data objects.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with condition and substitution columns.
        condition_col : str
            Column with condition names.
        substitutions_col : str
            Column with substitution strings.

        Returns
        -------
        dict[str, tuple]
            Mapping of condition -> (jaxmodels.Data, condition_df) for each
            condition present in df.
        """
        import scipy.sparse
        from jax.experimental import sparse as jsparse

        ref_bmap = self._data.binarymaps[self._data.reference]
        result = {}

        for condition, condition_df in df.groupby(condition_col):
            variant_subs = condition_df[substitutions_col]
            if condition not in self._data.reference_sequence_conditions:
                variant_subs = condition_df.apply(
                    lambda x: self._data.convert_subs_wrt_ref_seq(
                        condition, x[substitutions_col]
                    ),
                    axis=1,
                )

            row_ind = []
            col_ind = []
            unseen_mutations = set()

            for ivariant, subs in enumerate(variant_subs):
                try:
                    for isub in ref_bmap.sub_str_to_indices(subs):
                        row_ind.append(ivariant)
                        col_ind.append(isub)
                except ValueError:
                    if subs:
                        for mut in subs.split():
                            if mut not in self._data.mutations:
                                unseen_mutations.add(mut)

            if unseen_mutations:
                raise ValueError(
                    f"Variants contain mutations not seen during training: "
                    f"{sorted(unseen_mutations)}"
                )

            X = jsparse.BCOO.from_scipy_sparse(
                scipy.sparse.csr_matrix(
                    (np.ones(len(row_ind), dtype="int8"), (row_ind, col_ind)),
                    shape=(len(condition_df), ref_bmap.binarylength),
                    dtype="int8",
                )
            )

            x_wt = self._jax_data_sets[condition].x_wt
            func_scores = np.zeros(len(condition_df))
            if "func_score" in condition_df.columns:
                func_scores = condition_df["func_score"].values

            temp_data = jaxmodels.Data(
                x_wt=x_wt,
                X=X,
                functional_scores=func_scores,
            )
            result[condition] = (temp_data, condition_df)

        return result

    @property
    def training_loss(self) -> dict:
        """Per-condition and total loss on training data.

        Returns
        -------
        dict[str, float]
            Dictionary mapping condition names and ``"total"`` to their
            training loss values.

        Raises
        ------
        ValueError
            If model has not been fitted.
        """
        if self._jax_model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        loss_dict = self._loss_fn(
            self._jax_model, self._jax_data_sets, **self._loss_kwargs
        )
        result = {k: float(v) for k, v in loss_dict.items()}
        result["total"] = sum(result.values())
        return result

    def eval_loss(self, df):
        """Evaluate the model's loss on an arbitrary DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns 'condition', 'aa_substitutions',
            'func_score'.

        Returns
        -------
        dict[str, float]
            Per-condition losses and ``"total"`` loss.

        Raises
        ------
        ValueError
            If model is not fitted, required columns are missing,
            conditions are invalid, or substitutions contain unseen mutations.
        """
        if self._jax_model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        for col in ["condition", "aa_substitutions", "func_score"]:
            if col not in df.columns:
                raise ValueError(f"`df` lacks column '{col}'")

        invalid_conditions = set(df["condition"]) - set(self._data.conditions)
        if invalid_conditions:
            raise ValueError(
                f"Invalid conditions in df: {invalid_conditions}. "
                f"Valid conditions: {self._data.conditions}"
            )

        encoded = self._encode_variants(df)
        temp_data_sets = {condition: data for condition, (data, _) in encoded.items()}

        loss_dict = self._loss_fn(self._jax_model, temp_data_sets, **self._loss_kwargs)

        result = {k: float(v) for k, v in loss_dict.items()}
        result["total"] = sum(result.values())
        return result

    def add_phenotypes_to_df(
        self,
        df: pd.DataFrame,
        substitutions_col: str = "aa_substitutions",
        condition_col: str = "condition",
        predicted_phenotype_col: str = "predicted_func_score",
        overwrite_cols: bool = False,
    ) -> pd.DataFrame:
        """
        Add model predictions to a DataFrame of variants.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns specified by `condition_col` and
            `substitutions_col`. Additional columns will be preserved in output.
        substitutions_col : str
            Column in `df` giving variants as substitution strings.
            Default is 'aa_substitutions'.
        condition_col : str
            Column in `df` giving the condition for each variant.
            Values must exist in the model's conditions. Default is 'condition'.
        predicted_phenotype_col : str
            Name of column to add containing predicted functional scores.
            Default is 'predicted_func_score'.
        overwrite_cols : bool
            If the specified predicted phenotype column already exists in `df`,
            overwrite it? If False, raise an error.

        Returns
        -------
        pd.DataFrame
            A copy of `df` with predictions added. Always includes:

            - ``predicted_func_score`` (or custom name): predicted functional score
            - ``predicted_latent``: latent phenotype ``φ(X)``
            - ``predicted_fitness``: predicted fitness in ``g(φ)`` space

            If ``func_score`` column is present in `df`, also includes:

            - ``measured_fitness``: measured fitness in ``g(φ)`` space

        Raises
        ------
        ValueError
            If model is not fitted, required columns are missing, indices are
            not unique, conditions are invalid, or substitutions contain
            mutations not seen during training.

        Example
        -------
        >>> import pandas as pd
        >>> from multidms import Data, Model
        >>> df_train = pd.DataFrame({
        ...     'condition': ['a', 'a', 'b', 'b'],
        ...     'aa_substitutions': ['', 'M1A', '', 'M1A'],
        ...     'func_score': [0.0, 1.2, 0.1, 1.5]
        ... })
        >>> data = Data(df_train, reference='a')  # doctest: +ELLIPSIS
        >>> model = Model(data, ge_type='Identity', l2reg=0.01)
        >>> _ = model.fit(maxiter=5, warmstart=False, verbose=False)
        >>> df_new = pd.DataFrame({
        ...     'condition': ['a', 'b'],
        ...     'aa_substitutions': ['M1A', 'M1A']
        ... })
        >>> result = model.add_phenotypes_to_df(df_new)
        >>> 'predicted_func_score' in result.columns
        True
        >>> 'predicted_latent' in result.columns
        True
        >>> 'predicted_fitness' in result.columns
        True
        >>> len(result)
        2
        """
        if self._jax_model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        # Validate input
        if substitutions_col not in df.columns:
            raise ValueError(f"`df` lacks column '{substitutions_col}'")
        if condition_col not in df.columns:
            raise ValueError(f"`df` lacks column '{condition_col}'")
        if not df.index.is_unique:
            raise ValueError("`df` must have unique indices")

        # Check for invalid conditions
        invalid_conditions = set(df[condition_col]) - set(self._data.conditions)
        if invalid_conditions:
            raise ValueError(
                f"Invalid conditions in df: {invalid_conditions}. "
                f"Valid conditions: {self._data.conditions}"
            )

        # Return copy
        ret = df.copy()

        # Check if column exists and handle overwrite
        if predicted_phenotype_col in ret.columns and not overwrite_cols:
            raise ValueError(
                f"`df` already contains column '{predicted_phenotype_col}'. "
                "Set overwrite_cols=True to overwrite."
            )

        # Initialize prediction columns
        ret[predicted_phenotype_col] = np.nan
        ret["predicted_latent"] = np.nan
        ret["predicted_fitness"] = np.nan

        # Encode variants and predict
        encoded = self._encode_variants(
            df,
            condition_col=condition_col,
            substitutions_col=substitutions_col,
        )
        for condition, (temp_data, condition_df) in encoded.items():
            temp_data_sets = {condition: temp_data}
            predictions = self._jax_model.predict_score(temp_data_sets)

            phenotype_predictions = np.array(predictions[condition])
            assert len(phenotype_predictions) == len(condition_df)

            ret.loc[
                condition_df.index.values, predicted_phenotype_col
            ] = phenotype_predictions

            # Latent phenotype
            φ = self._jax_model.φ[condition]
            φ_X = np.array(φ(temp_data.X))
            ret.loc[condition_df.index.values, "predicted_latent"] = φ_X

            # Fitness in g(φ) space
            _α = self._jax_model.α
            α = float(_α[condition] if isinstance(_α, dict) else _α)
            g_φ_wt = float(self._jax_model.global_epistasis(φ(temp_data.x_wt)))
            ret.loc[condition_df.index.values, "predicted_fitness"] = (
                phenotype_predictions / α + g_φ_wt
            )

            # Measured fitness (only if func_score column exists)
            func_score_col = "func_score"
            if func_score_col in ret.columns:
                if "measured_fitness" not in ret.columns:
                    ret["measured_fitness"] = np.nan
                measured_scores = ret.loc[
                    condition_df.index.values, func_score_col
                ].values
                ret.loc[condition_df.index.values, "measured_fitness"] = (
                    measured_scores / α + g_φ_wt
                )

        return ret

    def get_ge_landscape_df(
        self,
        n_curve_points: int = 200,
    ) -> tuple:
        """Get data for plotting the global epistasis landscape.

        Returns a tuple of (variants_df, ge_curve_df). The variants DataFrame
        contains per-variant latent phenotype and fitness columns from
        :meth:`get_variants_df`, plus a ``wildtype_latent`` column for
        reference-line plotting. The curve DataFrame contains the global
        epistasis function evaluated over the observed latent phenotype range.

        Parameters
        ----------
        n_curve_points : int
            Number of points for the ``g(φ)`` curve grid.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(variants_df, ge_curve_df)`` where:

            - ``variants_df`` has all columns from :meth:`get_variants_df`
              plus ``wildtype_latent``.
            - ``ge_curve_df`` has columns ``predicted_latent`` and
              ``ge_curve_value``.
        """
        variants_df = self.get_variants_df()

        # Compute ge curve spanning the observed latent range
        φ_min = variants_df["predicted_latent"].min()
        φ_max = variants_df["predicted_latent"].max()
        margin = (φ_max - φ_min) * 0.05
        ge_curve = self.get_ge_curve(
            grid_min=float(φ_min - margin),
            grid_max=float(φ_max + margin),
            n_points=n_curve_points,
        )
        ge_curve = ge_curve.rename(
            columns={
                "latent": "predicted_latent",
                "observed": "ge_curve_value",
            }
        )

        # Add wildtype latent to variants_df for reference lines
        wt_latent = self.wildtype_latent
        variants_df["wildtype_latent"] = variants_df["condition"].map(wt_latent)

        return variants_df, ge_curve

    def plot_ge_landscape(self, n_curve_points=200, **kwargs):
        """Plot the global epistasis landscape.

        Convenience wrapper that calls :meth:`get_ge_landscape_df` and
        delegates to :func:`multidms.plot.ge_landscape`.

        Parameters
        ----------
        n_curve_points : int
            Number of points for the ``g(φ)`` curve grid. Default 200.
        **kwargs
            Passed to :func:`multidms.plot.ge_landscape`.

        Returns
        -------
        alt.LayerChart
            Interactive Altair chart.
        """
        import multidms.plot

        variants_df, ge_curve_df = self.get_ge_landscape_df(
            n_curve_points=n_curve_points
        )
        return multidms.plot.ge_landscape(variants_df, ge_curve_df, **kwargs)

    def get_ge_curve(
        self,
        grid_min: float = -5.0,
        grid_max: float = 5.0,
        n_points: int = 200,
    ) -> pd.DataFrame:
        """Evaluate the global epistasis function over a latent phenotype grid.

        Parameters
        ----------
        grid_min : float
            Minimum latent phenotype value for the grid.
        grid_max : float
            Maximum latent phenotype value for the grid.
        n_points : int
            Number of points in the grid.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'latent' and 'observed'.
        """
        if self._jax_model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        import jax.numpy as jnp

        grid = jnp.linspace(grid_min, grid_max, n_points)
        observed = self._jax_model.global_epistasis(grid)
        return pd.DataFrame(
            {
                "latent": np.array(grid),
                "observed": np.array(observed),
            }
        )

    @property
    def wildtype_latent(self) -> dict:
        """Wildtype latent phenotype for each condition.

        Returns
        -------
        dict[str, float]
            Dictionary mapping condition names to the wildtype's latent
            phenotype value in that condition.
        """
        if self._jax_model is None:
            raise ValueError("Model has not been fitted. Call fit() first.")
        result = {}
        for condition in self._data.conditions:
            latent = self._jax_model.φ[condition]
            x_wt = self._jax_data_sets[condition].x_wt
            result[condition] = float(latent.β0 + x_wt @ latent.β)
        return result

    def __repr__(self):
        """String representation."""
        return f"Model(ge_type='{self._ge_type}', loss_type='{self._loss_type}')"

    def __str__(self):
        """Detailed string representation."""
        fitted = "fitted" if self._jax_model is not None else "not fitted"
        return (
            f"Model\n"
            f"  ge_type: {self._ge_type}\n"
            f"  loss_type: {self._loss_type}\n"
            f"  l2reg: {self._l2reg}\n"
            f"  fusionreg: {self._fusionreg}\n"
            f"  status: {fitted}\n"
            f"  conditions: {self._data.conditions if self._data else None}\n"
        )
