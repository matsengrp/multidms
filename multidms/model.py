r"""
=====
model
=====

Defines :class:`Model` objects.
"""

import math
import warnings
from functools import partial, reduce

import jax
import jax.numpy as jnp
import numpy as onp
import pandas
import scipy
import seaborn as sns
from frozendict import frozendict
from jax.experimental import sparse
from jaxopt import ProximalGradient
from jaxopt.linear_solve import solve_normal_cg
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

import multidms.plot
from multidms import Data
import multidms.biophysical
from multidms.utils import is_wt


class Model:
    r"""
    Represent one or more DMS experiments
    to obtain tuned parameters that provide insight into
    individual mutational effects and conditional shifts
    of those effects on all non-reference conditions.
    For more see the biophysical model documentation

    Parameters
    ----------
    data : multidms.Data
        A reference to the dataset which will define the parameters
        of the model to be fit.
    epistatic_model : <class 'function'>
        A function which will transform the latent
        effects of mutations into a functional score.
        See the biophysical model documentation
        for more.
    output_activation : <class 'function'>
        A function which will transform the output of the
        global epistasis function. Defaults to the identity function
        (no activation). See the biophysical model documentation
    conditional_shifts : bool
        If true (default) initialize and fit the shift
        parameters for each non-reference condition.
        See Model Description section for more.
        Defaults to True.
    alpha_d : bool
        If True introduce a latent offset parameter
        for each condition.
        See the biophysical docs section for more.
        Defaults to True.
    gamma_corrected : bool
        If true (default), introduce the 'gamma' parameter
        for each non-reference parameter to
        account for differences between wild type
        behavior relative to its variants. This
        is essentially a bias added to the functional
        scores during fitting.
        See Model Description section for more.
        Defaults to False.
    PRNGKey : int
        The initial seed key for random parameters
        assigned to Betas and any other randomly
        initialized parameters.
        for more.
    init_beta_naught : float
        Initialize the latent offset parameter
        applied to all conditions.
        See the biophysical docs section for more.
    init_theta_scale : float
        Initialize the scaling parameter :math:`\theta_{\text{scale}}` of
        a two-parameter epistatic model (Sigmoid or Softplus).
    init_theta_bias : float
        Initialize the bias parameter :math:`\theta_{\text{bias}}` of
        a two parameter epistatic model (Sigmoid or Softplus).
    n_hidden_units : int or None
        If using `biophysical.multidms.biophysical.nn_global_epistasis`
        as the epistatic model, this is the number of hidden units
        used in the transform.
    lower_bound : float or None
        If using `biophysical.multidms.biophysical.softplus_activation`
        as the output activation, this is the lower bound of the
        softplus function.

    Example
    -------
    To create a ``Model`` object, all you need is
    the respective ``Data`` object for parameter fitting.

    >>> model = multidms.Model(data)

    Upon initialization, you will now have access to the underlying data
    and parameters.

    >>> model.data.mutations
    ('M1E', 'M1W', 'G3P', 'G3R')
    >>> model.data.conditions
    ('1', '2')
    >>> model.data.reference
    '1'
    >>> model.data.condition_colors
    {'1': '#0072B2', '2': '#009E73'}

    The mutations_df and variants_df may of course also be accessed.

    >>> model.data.mutations_df
      mutation wts  sites muts  times_seen_1  times_seen_2
    0      M1E   M      1    E             1           3.0
    1      M1W   M      1    W             1           0.0
    2      G3P   G      3    P             1           1.0
    3      G3R   G      3    R             1           2.0

    However, if accessed directly through the Model object, you will
    get the same information, along with model/parameter specific
    features included. These are automatically updated each time you
    request the property.

    >>> model.mutations_df
      mutation wts  sites muts  times_seen_1  ... beta  shift_1  F_1        shift_2   F_2
    0      M1E   M      1    E             1  ...  0.080868  0.0 -0.030881  0.0 -0.030881
    1      M1W   M      1    W             1  ... -0.386247  0.0 -0.049086  0.0 -0.049086
    2      G3P   G      3    P             1  ... -0.375656  0.0 -0.048574  0.0 -0.048574
    3      G3R   G      3    R             1  ...  1.668974  0.0 -0.006340  0.0 -0.006340

    Notice the respective single mutation effects ("beta"), conditional shifts (shift_d),
    and predicted functional score (F_d) of each mutation in the model are now
    easily accessible. Similarly, we can take a look at the variants_df for the model,

    >>> model.variants_df
      condition aa_substitutions  ...  predicted_func_score  corrected_func_score
    0         1              G3P  ...             -0.048574                  -0.5
    1         1              G3R  ...             -0.006340                  -7.0
    2         1              M1E  ...             -0.030881                   2.0
    3         1              M1W  ...             -0.049086                   2.3
    4         2              M1E  ...             -0.044834                   1.0
    5         2          M1E P3G  ...             -0.030881                   2.7
    6         2          M1E P3R  ...             -0.005848                  -2.7
    8         2              P3G  ...             -0.033464                   0.4
    9         2              P3R  ...             -0.006340                  -5.0

    We now have access to the predicted (and gamma corrected) functional scores
    as predicted by the models current parameters.

    So far, these parameters and predictions results from them have not been tuned
    to the dataset. Let's take a look at the loss on the training dataset
    given our initialized parameters

    >>> model.loss
    DeviceArray(4.40537408, dtype=float64)

    Next, we fit the model with some chosen hyperparameters.

    >>> model.fit(maxiter=1000, lasso_shift=1e-5)
    >>> model.loss
    DeviceArray(1.01582782, dtype=float64)

    The model tunes its parameters in place, and the subsequent call to retrieve
    the loss reflects our models loss given its updated parameters.
    """

    def __init__(
        self,
        data: Data,
        epistatic_model=multidms.biophysical.sigmoidal_global_epistasis,
        output_activation=multidms.biophysical.identity_activation,
        conditional_shifts=True,
        alpha_d=True,
        gamma_corrected=False,
        PRNGKey=0,
        init_beta_naught=0.0,
        init_theta_scale=5.0,
        init_theta_bias=-5.0,
        n_hidden_units=5,
        lower_bound=None,
    ):
        """See class docstring."""
        self.gamma_corrected = gamma_corrected
        self.conditional_shifts = conditional_shifts
        self.alpha_d = alpha_d

        self._data = data

        self._params = {}
        key = jax.random.PRNGKey(PRNGKey)

        # initialize beta and shift parameters
        # note that the only option is the additive model
        # as defined in multidms.biophysical.additive_model
        latent_model = multidms.biophysical.additive_model
        if latent_model == multidms.biophysical.additive_model:
            n_beta_shift = len(self._data.mutations)
            self._params["beta"] = jax.random.normal(shape=(n_beta_shift,), key=key)
            for condition in data.conditions:
                self._params[f"shift_{condition}"] = jnp.zeros(shape=(n_beta_shift,))
                self._params[f"alpha_{condition}"] = jnp.zeros(shape=(1,))
            self._params["beta_naught"] = jnp.array([init_beta_naught])
        else:
            raise ValueError(f"{latent_model} not recognized,")

        # initialize theta parameters
        if epistatic_model == multidms.biophysical.sigmoidal_global_epistasis:
            self._params["theta"] = dict(
                ge_scale=jnp.array([init_theta_scale]),
                ge_bias=jnp.array([init_theta_bias]),
            )

        elif epistatic_model == multidms.biophysical.softplus_global_epistasis:
            if output_activation != multidms.biophysical.softplus_activation:
                warnings.warn(
                    "softplus_global_epistasis has no natural lower bound,"
                    " we highly suggest using a softplus output activation"
                    "with a lower bound specified when using this model."
                )

            self._params["theta"] = dict(
                ge_scale=jnp.array([init_theta_scale]),
                ge_bias=jnp.array([init_theta_bias]),
            )

        elif epistatic_model == multidms.biophysical.identity_activation:
            self._params["theta"] = dict(ghost_param=jnp.zeros(shape=(1,)))

        elif epistatic_model == multidms.biophysical.nn_global_epistasis:
            key, key1, key2, key3, key4 = jax.random.split(key, num=5)
            self._params["theta"] = dict(
                p_weights_1=jax.random.normal(shape=(n_hidden_units,), key=key1).clip(
                    0
                ),
                p_weights_2=jax.random.normal(shape=(n_hidden_units,), key=key2).clip(
                    0
                ),
                p_biases=jax.random.normal(shape=(n_hidden_units,), key=key3),
                output_bias=jax.random.normal(shape=(1,), key=key4),
            )

        else:
            raise ValueError(f"{epistatic_model} not recognized,")

        if output_activation == multidms.biophysical.softplus_activation:
            if lower_bound is None:
                raise ValueError(
                    "softplus activation requires a lower bound be specified"
                )
            elif type(lower_bound) != float:
                raise ValueError("lower_bound must be a float")
            else:
                output_activation = partial(
                    multidms.biophysical.softplus_activation, lower_bound=lower_bound
                )

        for condition in data.conditions:
            self._params[f"gamma_{condition}"] = jnp.zeros(shape=(1,))

        # compile the model components
        pred = partial(
            multidms.biophysical._abstract_epistasis,  # abstract function to compile
            latent_model,
            epistatic_model,
            output_activation,
        )
        from_latent = partial(
            multidms.biophysical._abstract_from_latent,
            epistatic_model,
            output_activation,
        )
        cost = partial(multidms.biophysical._gamma_corrected_cost_smooth, pred)

        self._model_components = frozendict(
            {
                "latent_model": multidms.biophysical.additive_model,
                "g": epistatic_model,
                "output_activation": output_activation,
                "f": pred,
                "from_latent": from_latent,
                "objective": cost,
                "proximal": multidms.biophysical._lasso_lock_prox,
            }
        )

    @property
    def params(self) -> dict:
        """All current model parameters in a dictionary."""
        return self._params

    @property
    def data(self) -> multidms.Data:
        """
        multidms.Data Object this model references for fitting
        its parameters.
        """
        return self._data

    @property
    def model_components(self) -> frozendict:
        """
        A frozendict which hold the individual components of the model
        as well as the objective and forward functions.
        """
        return self._model_components

    @property
    def loss(self) -> float:
        """
        Compute model loss on all experimental training data
        without ridge or lasso penalties included.
        """
        kwargs = {
            "scale_coeff_ridge_beta": 0.0,
            "scale_coeff_ridge_shift": 0.0,
            "scale_coeff_ridge_gamma": 0.0,
        }
        data = (self.data.training_data["X"], self.data.training_data["y"])
        return jax.jit(self.model_components["objective"])(self.params, data, **kwargs)

    @property
    def variants_df(self):
        """
        Kept for backwards compatibility but will be removed in future versions.
        Please use `get_variants_df` instead.
        """
        warnings.warn("deprecated", DeprecationWarning)
        return self.get_variants_df(phenotype_as_effect=False)

    def get_variants_df(self, phenotype_as_effect=True):
        """
        Training data with model predictions for latent,
        and functional score phenotypes.

        Parameters
        ----------
        phenotype_as_effect : bool
            if True, phenotypes (both latent, and func_score)
            are calculated as the _difference_ between predicted
            phenotype of a given variant and the respective experimental
            wildtype prediction. Otherwise, report the unmodified
            model prediction.

        Returns
        -------
        pandas.DataFrame
            A copy of the training data, `self.data.variants_df`,
            with the phenotypes added. Phenotypes are predicted
            based on the current state of the model.
        """
        # this is what well update and return
        variants_df = self._data.variants_df.copy()

        # initialize new columns
        for pheno in ["latent", "func_score"]:
            variants_df[f"predicted_{pheno}"] = onp.nan

        # if we're a gamma corrected model, also report the "corrected"
        # observed func score, as we do during training.
        if self.gamma_corrected:
            variants_df["corrected_func_score"] = variants_df["func_score"]

        # get the wildtype predictions for each condition
        if phenotype_as_effect:
            wildtype_df = self.wildtype_df

        models = {
            "latent": jax.jit(self.model_components["latent_model"]),
            "func_score": jax.jit(self.model_components["f"]),
        }

        for condition, condition_df in variants_df.groupby("condition"):
            d_params = self.get_condition_params(condition)
            X = self._data.training_data["X"][condition]

            # prediction and effect
            for pheno in ["latent", "func_score"]:
                Y_pred = onp.array(models[pheno](d_params, X))
                if phenotype_as_effect:
                    Y_pred -= wildtype_df.loc[condition, f"predicted_{pheno}"]

                variants_df.loc[condition_df.index, f"predicted_{pheno}"] = Y_pred

            if self.gamma_corrected:
                variants_df.loc[condition_df.index, "corrected_func_score"] += d_params[
                    "gamma_d"
                ]

        return variants_df

    @property
    def mutations_df(self):
        """
        Kept for backwards compatibility but will be removed in future versions.
        Please use `get_mutations_df` instead.
        """
        warnings.warn("deprecated", DeprecationWarning)
        return self.get_mutations_df(phenotype_as_effect=False)

    def get_mutations_df(self, phenotype_as_effect=True):
        """
        Mutation attributes and phenotypic effects.

        Parameters
        ----------
        phenotype_as_effect : bool
            if True, phenotypes (both latent, and func_score)
            are calculated as the _difference_ between predicted
            phenotype of a given variant and the respective experimental
            wildtype prediction. Otherwise, report the unmodified
            model prediction.

        Returns
        -------
        pandas.DataFrame
            A copy of the mutations data, `self.data.mutations_df`,
            with the phenotypes added. Phenotypes are predicted
            based on the current state of the model.
        """
        # we're updating this
        mutations_df = self.data.mutations_df.copy()

        # for effect calculation
        if phenotype_as_effect:
            wildtype_df = self.wildtype_df

        # add betas i.e. 'latent effect'
        mutations_df.loc[:, "beta"] = self._params["beta"]
        X = sparse.BCOO.fromdense(onp.identity(len(self._data.mutations)))
        for condition in self._data.conditions:
            # shift of latent effect
            if condition != self._data.reference:
                mutations_df[f"shift_{condition}"] = self._params[f"shift_{condition}"]

            Y_pred = self.phenotype_frombinary(X, condition)
            if phenotype_as_effect:
                Y_pred -= wildtype_df.loc[condition, "predicted_func_score"]
            mutations_df["predicted_func_score"] = Y_pred

        return mutations_df

    def add_phenotypes_to_df(
        self,
        df,
        substitutions_col="aa_substitutions",
        condition_col="condition",
        latent_phenotype_col="predicted_latent",
        observed_phenotype_col="predicted_func_score",
        converted_substitutions_col="aa_subs_wrt_ref",
        overwrite_cols=False,
        unknown_as_nan=False,
        phenotype_as_effect=True,
    ):
        """Add predicted phenotypes to data frame of variants.

        Parameters
        ----------
        df : pandas.DataFrame
            Data frame containing variants. Requirements are the same as
            those used to initialize the `multidms.Data` object
        substitutions_col : str
            Column in `df` giving variants as substitution strings
            with respect to a given variants condition.
            These will be converted to be with respect to the reference sequence
            prior to prediction. Defaults to 'aa_substitutions'.
        condition_col : str
            Column in `df` giving the condition from which a variant was
            observed. Values must exist in the self.data.conditions and
            and error will be raised otherwise. Defaults to 'condition'.
        latent_phenotype_col : str
            Column added to `df` containing predicted latent phenotypes.
        observed_phenotype_col : str
            Column added to `df` containing predicted observed phenotypes.
        converted_substitutions_col : str or None
            Columns added to `df` containing converted substitution strings
            for non-reference conditions if they do not share a wildtype seq.
        overwrite_cols : bool
            If the specified latent or observed phenotype column already
            exist in `df`, overwrite it? If `False`, raise an error.
        unknown_as_nan : bool
            If some of the substitutions in a variant are not present in
            the model (not in :attr:`AbstractEpistasis.binarymap`) set the
            phenotypes to `nan` (not a number)? If `False`, raise an error.
        phenotype_as_effect : bool
            if True, phenotypes (both latent, and func_score)
            are calculated as the _difference_ between predicted
            phenotype of a given variant and the respective experimental
            wildtype prediction. Otherwise, report the unmodified
            model prediction.

        Returns
        -------
        pandas.DataFrame
            A copy of `df` with the phenotypes added. Phenotypes are predicted
            based on the current state of the model.
        """
        ref_bmap = self.data.binarymaps[self.data.reference]
        if substitutions_col is None:
            substitutions_col = ref_bmap.substitutions_col
        if substitutions_col not in df.columns:
            raise ValueError("`df` lacks `substitutions_col` " f"{substitutions_col}")
        if condition_col not in df.columns:
            raise ValueError("`df` lacks `condition_col` " f"{condition_col}")

        # return copy
        ret = df.copy()

        if phenotype_as_effect:
            wildtype_df = self.wildtype_df

        # initialize new columns
        for col in [
            latent_phenotype_col,
            observed_phenotype_col,
            converted_substitutions_col,
        ]:
            if col is None:
                continue
            if col in df.columns and not overwrite_cols:
                raise ValueError(f"`df` already contains column {col}")
            ret[col] = onp.nan

        for condition, condition_df in df.groupby(condition_col):
            variant_subs = condition_df[substitutions_col]
            if condition not in self.data.reference_sequence_conditions:
                variant_subs = condition_df.parallel_apply(
                    lambda x: self.data.convert_subs_wrt_ref_seq(
                        condition, x[substitutions_col]
                    ),
                    axis=1,
                )

            if converted_substitutions_col is not None:
                ret.loc[condition_df.index, converted_substitutions_col] = variant_subs

            # build binary variants as csr matrix, make prediction, and append
            row_ind = []  # row indices of elements that are one
            col_ind = []  # column indices of elements that are one
            nan_variant_indices = []  # indices of variants that are nan

            for ivariant, subs in enumerate(variant_subs):
                try:
                    for isub in ref_bmap.sub_str_to_indices(subs):
                        row_ind.append(ivariant)
                        col_ind.append(isub)
                except ValueError:
                    if unknown_as_nan:
                        nan_variant_indices.append(ivariant)
                    else:
                        raise ValueError(
                            "Variant has substitutions not in model:"
                            f"\n{subs}\nMaybe use `unknown_as_nan`?"
                        )

            X = sparse.BCOO.from_scipy_sparse(
                scipy.sparse.csr_matrix(
                    (onp.ones(len(row_ind), dtype="int8"), (row_ind, col_ind)),
                    shape=(len(condition_df), ref_bmap.binarylength),
                    dtype="int8",
                )
            )

            # latent predictions on binary variants, X
            latent_predictions = onp.array(
                self.latent_frombinary(X, condition=condition)
            )
            assert len(latent_predictions) == len(condition_df)
            if phenotype_as_effect:
                latent_predictions -= wildtype_df.loc[condition, "predicted_latent"]
            latent_predictions[nan_variant_indices] = onp.nan
            ret.loc[
                condition_df.index.values, latent_phenotype_col
            ] = latent_predictions

            # func_score predictions on binary variants, X
            phenotype_predictions = onp.array(
                self.phenotype_frombinary(X, condition=condition)
            )
            assert len(phenotype_predictions) == len(condition_df)
            if phenotype_as_effect:
                phenotype_predictions -= wildtype_df.loc[
                    condition, "predicted_func_score"
                ]
            phenotype_predictions[nan_variant_indices] = onp.nan
            ret.loc[
                condition_df.index.values, observed_phenotype_col
            ] = phenotype_predictions

        return ret

    @property
    def wildtype_df(self):
        """
        Get a dataframe indexed by condition wildtype
        containing the prediction features for each.
        """
        wildtype_df = (
            pandas.DataFrame(index=self.data.conditions)
            .assign(predicted_latent=onp.nan)
            .assign(predicted_func_score=onp.nan)
        )
        for condition in self.data.conditions:
            for pheno, model in zip(
                ["latent", "func_score"],
                [self.latent_fromsubs, self.phenotype_fromsubs],
            ):
                wildtype_df.loc[condition, f"predicted_{pheno}"] = model("", condition)

        return wildtype_df

    def mutation_site_summary_df(self, agg_func=onp.mean, times_seen_threshold=0):
        """
        Get all single mutational attributes from self._data
        updated with all model specific attributes, then aggregate
        all numerical columns by "sites" using
        ``agg`` function. The mean values are given by default.
        """
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        mut_df = self.mutations_df.select_dtypes(include=numerics)
        times_seen_cols = [c for c in mut_df.columns if "times" in c]
        for c in times_seen_cols:
            mut_df = mut_df[mut_df[c] >= times_seen_threshold]

        return mut_df.groupby("sites").aggregate(agg_func)

    def get_condition_params(self, condition=None):
        """Get the relent parameters for a model prediction"""
        condition = self.data.reference if condition is None else condition
        if condition not in self.data.conditions:
            raise ValueError(f"condition {condition} does not exist in model")

        return {
            "theta": self.params["theta"],
            "beta_m": self.params["beta"],
            "beta_naught": self.params["beta_naught"],
            "s_md": self.params[f"shift_{condition}"],
            "alpha_d": self.params[f"alpha_{condition}"],
            "gamma_d": self.params[f"gamma_{condition}"],
        }

    def phenotype_fromsubs(self, aa_subs, condition=None):
        """
        take a single string of subs which are
        not already converted wrt reference, convert them and
        then make a functional score prediction and return the result.
        """
        converted_subs = self.data.convert_subs_wrt_ref_seq(condition, aa_subs)
        X = jnp.array(
            [
                self.data.binarymaps[self.data.reference].sub_str_to_binary(
                    converted_subs
                )
            ]
        )
        return self.phenotype_frombinary(X)

    def latent_fromsubs(self, aa_subs, condition=None):
        """
        take a single string of subs which are
        not already converted wrt reference, convert them and
        then make a latent prediction and return the result.
        """
        converted_subs = self.data.convert_subs_wrt_ref_seq(condition, aa_subs)
        X = jnp.array(
            [
                self.data.binarymaps[self.data.reference].sub_str_to_binary(
                    converted_subs
                )
            ]
        )
        return self.latent_frombinary(X)

    def phenotype_frombinary(self, X, condition=None):
        """
        Condition specific functional score prediction
        on X using the biophysical model
        given current model parameters.

        Parameters
        ----------
        X : jnp.array
            Binary encoded variants to make predictions on.
        condition : str
            Condition to make predictions for. If None, use the reference
        """
        d_params = self.get_condition_params(condition)
        return jax.jit(self.model_components["f"])(d_params, X)

    def latent_frombinary(self, X, condition=None):
        """
        Condition specific latent phenotype prediction
        on X using the biophysical model
        given current model parameters.

        Parameters
        ----------
        X : jnp.array
            Binary encoded variants to make predictions on.
        condition : str
            Condition to make predictions for. If None, use the reference
        """
        d_params = self.get_condition_params(condition)
        return jax.jit(self.model_components["latent_model"])(d_params, X)

    def fit_reference_beta(self, **kwargs):
        """
        Fit the Model beta's to the reference data.

        This is an experimental feature and is not recommended
        for general use.
        """
        ref_X = self.data.training_data["X"][self.data.reference]
        ref_y = self.data.training_data["y"][self.data.reference]

        self._params["beta"] = solve_normal_cg(
            lambda beta: ref_X @ beta, ref_y, init=self._params["beta"], **kwargs
        )

    def fit(self, lasso_shift=1e-5, tol=1e-6, maxiter=1000, lock_params={}, **kwargs):
        r"""
        Use jaxopt.ProximalGradiant to optimize the model's free parameters.

        Parameters
        ----------
        lasso_shift : float
            L1 penalty on the shift parameters. Defaults to 1e-5.
        tol : float
            Tolerance for the optimization. Defaults to 1e-6.
        maxiter : int
            Maximum number of iterations for the optimization. Defaults to 1000.
        lock_params : dict
            Dictionary of parameters, and desired value to constrain
            them at during optimization. By default, none of the parameters
            besides the reference shift, and reference latent offset are locked.
        **kwargs : dict
            Additional keyword arguments passed to the objective function.
            These include hyperparameters like a ridge penalty on beta, shift, and gamma
            as well as huber loss scaling.
        """
        solver = ProximalGradient(
            jax.jit(self._model_components["objective"]),
            jax.jit(self._model_components["proximal"]),
            tol=tol,
            maxiter=maxiter,
        )

        lock_params[f"shift_{self._data.reference}"] = jnp.zeros(
            len(self._params["beta"])
        )
        lock_params[f"gamma_{self._data.reference}"] = jnp.zeros(shape=(1,))

        if not self.conditional_shifts:
            for condition in self._data.conditions:
                lock_params[f"shift_{condition}"] = jnp.zeros(shape=(1,))

        if not self.gamma_corrected:
            for condition in self._data.conditions:
                lock_params[f"gamma_{condition}"] = jnp.zeros(shape=(1,))

        if not self.alpha_d:
            for condition in self._data.conditions:
                lock_params[f"alpha_{condition}"] = jnp.zeros(shape=(1,))
        else:
            lock_params[f"alpha_{self._data.reference}"] = jnp.zeros(shape=(1,))

        lasso_params = {}
        for non_ref_condition in self._data.conditions:
            if non_ref_condition == self._data.reference:
                continue
            lasso_params[f"shift_{non_ref_condition}"] = lasso_shift

        self._params, state = solver.run(
            self._params,
            hyperparams_prox=dict(lasso_params=lasso_params, lock_params=lock_params),
            data=(self._data.training_data["X"], self._data.training_data["y"]),
            **kwargs,
        )

    def plot_pred_accuracy(
        self, hue=True, show=True, saveas=None, annotate_corr=True, ax=None, **kwargs
    ):
        """
        Create a figure which visualizes the correlation
        between model predicted functional score of all
        variants in the training with ground truth measurements.
        """
        df = self.variants_df

        df = df.assign(is_wt=df["aa_substitutions"].apply(is_wt))

        if ax is None:
            fig, ax = plt.subplots(figsize=[3, 3])

        func_score = "corrected_func_score" if self.gamma_corrected else "func_score"
        sns.scatterplot(
            data=df.sample(frac=1),
            x="predicted_func_score",
            y=func_score,
            hue="condition" if hue else None,
            palette=self.data.condition_colors,
            ax=ax,
            **kwargs,
        )

        for condition, values in self.wildtype_df.iterrows():
            ax.axvline(
                values.predicted_func_score,
                label=condition,
                c=self._data.condition_colors[condition],
                lw=2,
            )

        xlb, xub = [-1, 1] + onp.quantile(df.predicted_func_score, [0.05, 1.0])
        ylb, yub = [-1, 1] + onp.quantile(df[func_score], [0.05, 1.0])

        ax.plot([ylb, yub], [ylb, yub], "k--", lw=2)
        if annotate_corr:
            start_y = 0.95
            for c, cdf in df.groupby("condition"):
                r = pearsonr(cdf[func_score], cdf["predicted_func_score"])[0]
                ax.annotate(
                    f"$r = {r:.2f}$",
                    (0.01, start_y),
                    xycoords="axes fraction",
                    fontsize=12,
                    c=self._data.condition_colors[c],
                )
                start_y += -0.05
        ax.set_ylabel("functional score")
        ax.set_xlabel("predicted functional score")

        ax.axhline(0, color="k", ls="--", lw=2)
        ax.axvline(0, color="k", ls="--", lw=2)

        ax.set_ylabel("functional score + gamma$_{d}$")
        plt.tight_layout()
        if saveas:
            fig.savefig(saveas)
        if show:
            plt.show()
        return ax

    def plot_epistasis(
        self, hue=True, show=True, saveas=None, ax=None, sample=1.0, **kwargs
    ):
        """
        Plot latent predictions against
        gamma corrected ground truth measurements
        of all samples in the training set.
        """
        df = self.variants_df

        df = df.assign(is_wt=df["aa_substitutions"].apply(is_wt))

        if ax is None:
            fig, ax = plt.subplots(figsize=[3, 3])

        func_score = "corrected_func_score" if self.gamma_corrected else "func_score"
        sns.scatterplot(
            data=df.sample(frac=sample),
            x="predicted_latent",
            y=func_score,
            hue="condition" if hue else None,
            palette=self.data.condition_colors,
            ax=ax,
            **kwargs,
        )

        for condition, values in self.wildtype_df.iterrows():
            ax.axvline(
                values.predicted_latent,
                label=condition,
                c=self._data.condition_colors[condition],
                lw=2,
            )

        xlb, xub = [-1, 1] + onp.quantile(df.predicted_latent, [0.05, 1.0])
        ylb, yub = [-1, 1] + onp.quantile(df[func_score], [0.05, 1.0])

        latent_model_grid = onp.linspace(xlb, xub, num=1000)

        params = self.get_condition_params(self._data.reference)
        latent_preds = self._model_components["g"](params["theta"], latent_model_grid)
        shape = (latent_model_grid, latent_preds)
        ax.plot(*shape, color="k", lw=2)

        ax.axhline(0, color="k", ls="--", lw=2)
        ax.set_xlim([xlb, xub])
        ax.set_ylim([ylb, yub])
        ax.set_ylabel("functional score")
        ax.set_xlabel("predicted latent phenotype")
        plt.tight_layout()

        if saveas:
            fig.savefig(saveas)
        if show:
            plt.show()
        return ax

    def plot_param_hist(
        self, param, show=True, saveas=False, times_seen_threshold=3, ax=None, **kwargs
    ):
        """Plot the histogram of a parameter."""
        mut_effects_df = self.mutations_df

        if ax is None:
            fig, ax = plt.subplots(figsize=[3, 3])

        times_seen_cols = [c for c in mut_effects_df.columns if "times" in c]
        for c in times_seen_cols:
            mut_effects_df = mut_effects_df[mut_effects_df[c] >= times_seen_threshold]

        # Plot data for all mutations
        data = mut_effects_df[mut_effects_df["muts"] != "*"]
        bin_width = 0.25
        min_val = math.floor(data[param].min()) - 0.25 / 2
        max_val = math.ceil(data[param].max())
        sns.histplot(
            x=param,
            data=data,
            ax=ax,
            stat="density",
            label="muts to amino acids",
            binwidth=bin_width,
            binrange=(min_val, max_val),
            alpha=0.5,
            **kwargs,
        )

        # Plot data for mutations leading to stop codons
        data = mut_effects_df[mut_effects_df["muts"] == "*"]
        if len(data) > 0:
            sns.histplot(
                x=param,
                data=data,
                ax=ax,
                stat="density",
                label="muts to stop codons",
                binwidth=bin_width,
                binrange=(min_val, max_val),
                alpha=0.5,
                **kwargs,
            )

        # ax.set_yscale("log")

        ax.set(xlabel=param)
        plt.tight_layout()

        if saveas:
            fig.savefig(saveas)
        if show:
            plt.show()
        return ax

    def plot_param_heatmap(
        self, param, show=True, saveas=False, times_seen_threshold=3, ax=None, **kwargs
    ):
        """
        Plot the heatmap of a parameters
        associated with specific sites and substitutions.
        """
        if not param.startswith("beta") and not param.startswith("S"):
            raise ValueError(
                "Parameter to visualize must be an existing beta, or shift parameter"
            )

        mut_effects_df = self.mutations_df

        if ax is None:
            fig, ax = plt.subplots(figsize=[12, 3])

        times_seen_cols = [c for c in mut_effects_df.columns if "times" in c]
        for c in times_seen_cols:
            mut_effects_df = mut_effects_df[mut_effects_df[c] >= times_seen_threshold]

        mut_effects_df.sites = mut_effects_df.sites.astype(int)
        mutation_effects = mut_effects_df.pivot(
            index="muts", columns="sites", values=param
        )

        sns.heatmap(
            mutation_effects,
            mask=mutation_effects.isnull(),
            cmap="coolwarm_r",
            center=0,
            cbar_kws={"label": param},
            ax=ax,
            **kwargs,
        )

        plt.tight_layout()
        if saveas:
            fig.savefig(saveas)
        if show:
            plt.show()
        return ax

    def plot_shifts_by_site(
        self,
        condition,
        show=True,
        saveas=False,
        times_seen_threshold=3,
        agg_func=onp.mean,
        ax=None,
        **kwargs,
    ):
        """Summarize shift parameter values by associated sites and conditions."""
        mutation_site_summary_df = self.mutation_site_summary_df(
            agg_func, times_seen_threshold=times_seen_threshold
        ).reset_index()

        if ax is None:
            fig, ax = plt.subplots(figsize=[12, 3])

        ax.axhline(0, color="k", ls="--", lw=1)

        sns.lineplot(
            data=mutation_site_summary_df,
            x="sites",
            y=f"shift_{condition}",
            color=self._data.condition_colors[condition],
            ax=ax,
            # legend=True,
            **kwargs,
        )
        color = [
            self.data.condition_colors[condition]
            if s not in self.data.non_identical_sites[condition]
            else (0.0, 0.0, 0.0)
            for s in mutation_site_summary_df.sites
        ]
        size = [
            0.5 if s not in self.data.non_identical_sites[condition] else 1.0
            for s in mutation_site_summary_df.sites
        ]
        sns.scatterplot(
            data=mutation_site_summary_df,
            x="sites",
            y=f"shift_{condition}",
            size=size,
            color=onp.array(color),
            ax=ax,
            legend=False,
            **kwargs,
        )

        if show:
            plt.show()
        return ax

    def plot_fit_param_comp_scatter(
        self,
        other,
        self_param="beta",
        other_param="beta",
        figsize=[5, 4],
        saveas=None,
        show=True,
        site_agg_func=None,
    ):
        """Plot a scatter plot of the parameter values of two models"""
        if not site_agg_func:
            dfs = [self.mutations_df, other.mutations_df]
        else:
            dfs = [
                self.mutation_site_summary_df(agg=site_agg_func).reset_index(),
                other.mutation_site_summary_df(agg=site_agg_func).reset_index(),
            ]

        combine_on = "mutation" if site_agg_func is None else "sites"
        comb_mut_effects = reduce(
            lambda l, r: pandas.merge(l, r, how="inner", on=combine_on),  # noqa: E741
            dfs,
        )
        comb_mut_effects["is_stop"] = [
            True if "*" in s else False for s in comb_mut_effects[combine_on]
        ]

        same = self_param == other_param
        x = f"{self_param}_x" if same else self_param
        y = f"{other_param}_y" if same else other_param

        fig, ax = plt.subplots(figsize=figsize)
        r = pearsonr(comb_mut_effects[x], comb_mut_effects[y])[0]
        sns.scatterplot(
            data=comb_mut_effects,
            x=x,
            y=y,
            hue="is_stop",
            alpha=0.6,
            palette="deep",
            ax=ax,
        )

        xlb, xub = [-1, 1] + onp.quantile(comb_mut_effects[x], [0.00, 1.0])
        ylb, yub = [-1, 1] + onp.quantile(comb_mut_effects[y], [0.00, 1.0])
        min1 = min(xlb, ylb)
        max1 = max(xub, yub)
        ax.plot([min1, max1], [min1, max1], ls="--", c="k")
        ax.annotate(f"$r = {r:.2f}$", (0.7, 0.1), xycoords="axes fraction", fontsize=12)
        plt.tight_layout()

        if saveas:
            fig.saveas(saveas)
        if show:
            plt.show()

        return fig, ax

    def mut_shift_plot(self, **kwargs):
        """
        Wrapper method for visualizing the shift plot.
        see `multidms.plot.mut_shift_plot()` for more
        """  # noqa: D401
        return multidms.plot.mut_shift_plot(self, **kwargs)
