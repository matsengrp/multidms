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
        Ridge penalty for β0 offsets from reference condition (default: 0.0).

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

        # Will be populated by fit()
        self._jax_model = None
        self._jax_data_sets = None
        self._loss_trajectory = None

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
    def convergence_trajectory_df(self) -> pd.DataFrame:
        """
        Convergence trajectory showing loss over iterations.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'iteration', 'loss', 'error'
        """
        if self._loss_trajectory is None:
            return None

        df = pd.DataFrame(
            {
                "iteration": range(len(self._loss_trajectory)),
                "loss": self._loss_trajectory,
            }
        )
        # Calculate error as change in loss
        df["error"] = df["loss"].diff().abs().fillna(0.0)
        return df

    # See issue #178 for optimization of re-fitting already fitted models
    def fit(
        self,
        warmstart: bool = True,
        maxiter: int = 10,
        tol: float = 1e-6,
        beta0_init: dict = None,
        beta_init: dict = None,
        alpha_init: dict = None,
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
        alpha_init : dict, optional
            Initial α scaling values per condition.
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

        # Fit model using jaxmodels
        self._jax_model, self._loss_trajectory = jaxmodels.fit(
            data_sets=self._jax_data_sets,
            reference_condition=self._data.reference,
            l2reg=self._l2reg,
            fusionreg=self._fusionreg,
            beta0_ridge=self._beta0_ridge,
            block_iters=maxiter,
            block_tol=tol,
            global_epistasis=global_epistasis,
            loss_fn=loss_fn,
            warmstart=warmstart,
            beta0_init=beta0_init,
            beta_init=beta_init,
            alpha_init=alpha_init,
            beta_clip_range=beta_clip_range,
            ge_kwargs=ge_kwargs,
            cal_kwargs=cal_kwargs,
            loss_kwargs=loss_kwargs,
            verbose=verbose,
        )

        return self

    # See issue #179 for removal of deprecated phenotype_as_effect parameter
    def get_mutations_df(self, phenotype_as_effect: bool = True) -> pd.DataFrame:
        """
        Extract mutation-level parameters in wide format.

        Parameters
        ----------
        phenotype_as_effect : bool
            If True, report mutation effects. If False, report raw latent phenotypes.

        Returns
        -------
        pd.DataFrame
            DataFrame with mutations as rows (index) and columns:
            - beta_{condition} for each condition
            - shift_{condition} for each non-reference condition
            Shift parameters represent the difference in beta values between each
            condition and the reference condition.

        Example
        -------
        For a model with conditions ['a', 'b'] where 'a' is reference:
        - Columns: beta_a, beta_b, shift_b
        - One row per mutation
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

            result_rows.append(cond_data)

        return pd.concat(result_rows, ignore_index=True)

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
            A copy of `df` with predictions added.

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

        # Initialize prediction column
        ret[predicted_phenotype_col] = np.nan

        # Get reference binarymap for encoding
        ref_bmap = self._data.binarymaps[self._data.reference]

        # Process each condition separately
        for condition, condition_df in df.groupby(condition_col):
            # Convert substitutions to reference frame if needed
            variant_subs = condition_df[substitutions_col]
            if condition not in self._data.reference_sequence_conditions:
                variant_subs = condition_df.apply(
                    lambda x: self._data.convert_subs_wrt_ref_seq(
                        condition, x[substitutions_col]
                    ),
                    axis=1,
                )

            # Build binary variant matrix
            row_ind = []  # row indices of elements that are one
            col_ind = []  # column indices of elements that are one
            unseen_mutations = set()

            for ivariant, subs in enumerate(variant_subs):
                try:
                    for isub in ref_bmap.sub_str_to_indices(subs):
                        row_ind.append(ivariant)
                        col_ind.append(isub)
                except ValueError:
                    # Extract the individual mutations that are unseen
                    if subs:  # non-empty string
                        for mut in subs.split():
                            if mut not in self._data.mutations:
                                unseen_mutations.add(mut)

            # If there are unseen mutations, raise an error
            if unseen_mutations:
                raise ValueError(
                    f"Variants contain mutations not seen during training: "
                    f"{sorted(unseen_mutations)}"
                )

            # Create sparse matrix
            import scipy.sparse
            from jax.experimental import sparse as jsparse

            X = jsparse.BCOO.from_scipy_sparse(
                scipy.sparse.csr_matrix(
                    (np.ones(len(row_ind), dtype="int8"), (row_ind, col_ind)),
                    shape=(len(condition_df), ref_bmap.binarylength),
                    dtype="int8",
                )
            )

            # Create jaxmodels.Data object for this condition
            # We need x_wt from the training data
            x_wt = self._jax_data_sets[condition].x_wt

            # Create a temporary Data object with dummy functional scores
            import multidms.jaxmodels as jaxmodels

            temp_data = jaxmodels.Data(
                x_wt=x_wt,
                X=X,
                functional_scores=np.zeros(len(condition_df)),  # dummy values
            )

            # Make predictions using jaxmodels
            temp_data_sets = {condition: temp_data}
            predictions = self._jax_model.predict_score(temp_data_sets)

            # Extract predictions for this condition
            phenotype_predictions = np.array(predictions[condition])
            assert len(phenotype_predictions) == len(condition_df)

            # Add predictions to result dataframe
            ret.loc[
                condition_df.index.values, predicted_phenotype_col
            ] = phenotype_predictions

        return ret

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
