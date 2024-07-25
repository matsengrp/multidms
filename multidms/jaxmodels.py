r"""A simple API for global epistasis modeling."""

import multidms

import equinox as eqx
from jaxtyping import Array, Float, Int
import jax
import jax.numpy as jnp
from jax.scipy.special import expit, xlogy, gammaln
import jaxopt

import scipy
from sklearn import linear_model

jax.config.update("jax_enable_x64", True)


class Data(eqx.Module):
    r"""Data for a DMS experiment.

    Args:
        multidms_data: The data to use. Note: the WT must be the first variant in each condition.
        condition: The condition to extract data for.
    """

    x_wt: Int[Array, "n_mutations"]
    """Binary encoding of the wildtype sequence."""
    pre_count_wt: Float[Array, ""]
    """Wildtype pre-selection count."""
    post_count_wt: Float[Array, ""]
    """Wildtype post-selection count."""
    X: Int[Array, "n_variants n_mutations"]
    """Variant encoding matrix (sparse format)."""
    pre_counts: Int[Array, "n_variants"]
    """Pre-selection counts for each variant."""
    post_counts: Int[Array, "n_variants"]
    """Post-selection counts for each variant."""
    functional_scores: Float[Array, "n_variants"]
    """Functional scores for each variant."""

    def __init__(self, multidms_data: multidms.Data, condition: str) -> None:
        X = multidms_data.arrays["X"][condition]
        not_wt = X.indices[:, 0] != 0
        sparse_data = X.data[not_wt]
        sparse_idxs = X.indices[not_wt]
        sparse_idxs = sparse_idxs.at[:, 0].add(-1)
        X = jax.experimental.sparse.BCOO(
            (sparse_data, sparse_idxs), shape=(X.shape[0] - 1, X.shape[1])
        )

        self.x_wt = multidms_data.arrays["X"][condition][0].todense()
        self.pre_count_wt = multidms_data.arrays["pre_count"][condition][0]
        self.post_count_wt = multidms_data.arrays["post_count"][condition][0]
        self.X = X
        self.pre_counts = multidms_data.arrays["pre_count"][condition][1:]
        self.post_counts = multidms_data.arrays["post_count"][condition][1:]
        self.functional_scores = multidms_data.arrays["y"][condition][1:]


class Latent(eqx.Module):
    r"""Model a latent phenotype.

    Args:
        data: Data to initialize the model shape for.
        l2reg: L2 regularization strength for warmstart of latent models.
    """

    β0: Float[Array, ""]
    """Intercept."""
    β: Float[Array, "n_mutations"]
    """Mutation effects."""

    def __init__(
        self,
        data: Data,
        l2reg=0.0,
    ) -> None:
        X = scipy.sparse.csr_array(
            (data.X.data, data.X.indices.T), shape=(data.X.shape[0], len(data.x_wt))
        )
        y = data.functional_scores
        ridge_solver = linear_model.Ridge(alpha=l2reg)
        ridge_solver.fit(X, y, sample_weight=jnp.log(data.pre_counts))
        self.β0 = jnp.asarray(ridge_solver.intercept_)
        self.β = jnp.asarray(ridge_solver.coef_)

    def __call__(
        self,
        X: Float[Array, "n_variants n_mutations"],
    ) -> Float[Array, "n_variants"]:
        return self.β0 + X @ self.β

    def predict_latent(
        self,
        data: Data,
    ) -> Float[Array, "n_variants"]:
        return self(data.X)


class Model(eqx.Module):
    r"""Model DMS data.

    Args:
        φ: Latent models for each condition.
        α: fitness-functional score scaling factors for each condition.
        reference_condition: The condition to use as a reference.
    """

    φ: dict[str, Latent]
    """Latent models for each condition."""
    α: dict[str, Float[Array, ""]]
    """Fitness-functional score scaling factors for each condition."""
    reference_condition: str = eqx.field(static=True)
    """The condition to use as a reference."""

    def g(self, φ_val: Float[Array, "n_variants"]) -> Float[Array, "n_variants"]:
        r"""The global epistasis function.

        Args:
            φ_val: The latent phenotype.

        Returns:
            The fitness score for the given latent phenotype.
        """
        return expit(φ_val)

    def predict_latent(
        self,
        data_sets: dict[str, Data],
    ) -> dict[str, Float[Array, "n_variants"]]:
        r"""Predict latent phenotypes.

        Args:
            data_sets: Data sets for each condition.
        """
        result = {}
        for d in data_sets:
            φ = self.φ[d]
            X = data_sets[d].X
            result[d] = φ(X)
        return result

    def predict_score(
        self,
        data_sets: dict[str, Data],
    ) -> dict[str, Float[Array, "n_variants"]]:
        r"""Predict fitness-functional scores.

        Args:
            data_sets: Data sets for each condition.
        """
        result = {}
        for d in data_sets:
            φ = self.φ[d]
            α = self.α[d]
            X = data_sets[d].X
            x_wt = data_sets[d].x_wt
            result[d] = α * (self.g(φ(X)) - self.g(φ(x_wt)))
        return result

    def predict_post_count(
        self,
        data_sets: dict[str, Data],
    ) -> dict[str, Float[Array, "n_variants"]]:
        r"""Predict post-selection counts.

        Args:
            data_sets: Data sets for each condition.
        """
        result = {}
        score_pred = self.predict_score(data_sets)
        for d in data_sets:
            f = score_pred[d]
            n_v = data_sets[d].pre_counts
            n_wt = data_sets[d].pre_count_wt
            m_wt = data_sets[d].post_count_wt
            result[d] = jnp.power(
                2,
                f + jnp.log2(m_wt) - jnp.log2(n_wt) + jnp.log2(n_v),
            )
        return result

    def loss(
        self,
        data_sets: dict[str, Data],
    ) -> dict[str, Float[Array, ""]]:
        r"""Compute the loss.

        Args:
            data_sets: Data sets for each condition.
        """
        post_count_pred = self.predict_post_count(data_sets)
        result = {}
        for d in data_sets:
            m_v = data_sets[d].post_counts
            m_v_pred = post_count_pred[d]
            result[d] = (m_v_pred - xlogy(m_v, m_v_pred) + gammaln(m_v + 1)).sum()
        return result


def fit(
    data_sets: dict[str, Data],
    reference_condition: str,
    l2reg_α=0.0,
    l2reg=0.0,
    fusionreg=0.0,
    opt_kwargs=dict(tol=1e-8, maxiter=1000),
) -> tuple[Model, jaxopt._src.proximal_gradient.ProxGradState]:
    r"""
    Fit a model to data.

    Args:
        data_sets: Data to fit to. Each key is a condition.
        reference_condition: The condition to use as a reference.
        l2reg_α: L2 regularization strength for α.
        l2reg: L2 regularization strength for mutation effects.
        fusionreg: Fusion (shift lasso) regularization strength.
        opt_kwargs: Keyword arguments to pass to solver.

    Returns:
        Fitted model.
    """
    if data_sets[reference_condition].x_wt.sum() != 0:
        raise ValueError(
            "WT sequence of the reference condition should have no mutations."
        )

    opt = jaxopt.ProximalGradient(
        _objective_smooth_preconditioned,
        prox=_prox,
        value_and_grad=True,
        **opt_kwargs,
    )

    model = Model(
        φ={d: Latent(data_sets[d], l2reg=l2reg) for d in data_sets},
        α={d: jnp.ptp(data_sets[d].functional_scores) for d in data_sets},
        reference_condition=reference_condition,
    )

    Ps = {d: jnp.diag(1 / (1 + data_sets[d].X.sum(axis=0).todense())) for d in data_sets}

    hyperparameters = dict(fusionreg=fusionreg, Ps=Ps)
    args = (data_sets, Ps)
    kwargs = dict(l2reg_α=l2reg_α, l2reg=l2reg)

    model, state = opt.run(model, hyperparameters, *args, **kwargs)

    return model, state


# The following private functions are used internally by the fit function


@jax.jit
@jax.value_and_grad
def _objective_smooth(model, data_sets, l2reg_α=0.0, l2reg=0.0):
    n = sum(data_set.X.shape[0] for data_set in data_sets.values())
    loss = sum(model.loss(data_sets).values())
    ridge_α = sum((model.α[d] ** 2).sum() for d in model.φ)
    ridge_β = sum((model.φ[d].β ** 2).sum() for d in model.φ)
    return loss / n + l2reg_α * ridge_α + l2reg * ridge_β


@jax.jit
def _objective_smooth_preconditioned(model, data_sets, Ps, l2reg_α=0.0, l2reg=0.0):
    value, grad = _objective_smooth(model, data_sets, l2reg_α, l2reg)
    for d in model.φ:
        P = Ps[d]
        grad = eqx.tree_at(lambda grad_: grad_.φ[d].β, grad, P @ grad.φ[d].β)
    return value, grad


def _objective(model, data_sets, fusionreg=0.0, l2reg_α=0.0, l2reg=0.0):
    return _objective_smooth(
        model, data_sets, l2reg_α=l2reg_α, l2reg=l2reg
    ) + fusionreg * sum(
        jnp.abs(model.φ[d].β - model.φ[model.reference_condition].β).sum()
        for d in model.φ
        if d != model.reference_condition
    )


@jax.jit
def _prox(model, hyperparameters, scaling=1.0):
    fusionreg = hyperparameters["fusionreg"]
    Ps = hyperparameters["Ps"]
    # fused lasso for β
    β_ref = model.φ[model.reference_condition].β
    for d in model.φ:
        if d != model.reference_condition:
            β0 = model.φ[d].β0
            β = model.φ[d].β
            P = Ps[d]
            β_prox = β_ref + jaxopt.prox.prox_lasso(β - β_ref, fusionreg * jnp.diag(P), scaling)
            model = eqx.tree_at(lambda model_: model_.φ[d].β0, model, β0)
            model = eqx.tree_at(lambda model_: model_.φ[d].β, model, β_prox)
    # box constrain α to the non-negative orthant
    for d in model.φ:
        model = eqx.tree_at(
            lambda model: model.α[d], model, jnp.clip(model.α[d], 0.0, jnp.inf)
        )
    return model
