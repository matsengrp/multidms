r"""A simple API for global epistasis modeling."""

import multidms

import equinox as eqx
from jaxtyping import Array, Float
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

    x_wt: Float[Array, "n_mutations"]
    """Binary encoding of the wildtype sequence."""
    pre_count_wt: Float[Array, ""]
    """Wildtype pre-selection count."""
    post_count_wt: Float[Array, ""]
    """Wildtype post-selection count."""
    X: Float[Array, "n_variants n_mutations"]
    """Variant encoding matrix (sparse format)."""
    pre_counts: Float[Array, "n_variants"]
    """Pre-selection counts for each variant."""
    post_counts: Float[Array, "n_variants"]
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
        l2reg_β: L2 regularization strength for warmstart of latent models.
    """

    β0: Float[Array, ""]
    """Intercept."""
    β: Float[Array, "n_mutations"]
    """Mutation effects."""

    def __init__(
        self,
        data: Data,
        l2reg_β=0.0,
    ) -> None:
        X = scipy.sparse.csr_array(
            (data.X.data, data.X.indices.T), shape=(data.X.shape[0], len(data.x_wt))
        )
        y = data.functional_scores
        ridge_solver = linear_model.Ridge(alpha=l2reg_β)
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
    l2reg_β=0.0,
    fusionreg=0.0,
    maxiter=10,
    obj_tol=1e-4,
    block_kwargs=dict(tol=1e-8, maxiter=1000),
) -> tuple[Model, list[float]]:
    r"""
    Fit a model to data.

    Args:
        data_sets: Data to fit to. Each key is a condition.
        reference_condition: The condition to use as a reference.
        l2reg_α: L2 regularization strength for α.
        l2reg_β: L2 regularization strength for β.
        fusionreg: Fusion (shift lasso) regularization strength.
        maxiter: Maximum number of iterations of block coordinate descent.
        opt_tol: Optimization tolerance for block coordinate descent.
        block_kwargs: Keyword arguments to pass to the block solvers

    Returns:
        Fitted model.
    """
    if data_sets[reference_condition].x_wt.sum() != 0:
        raise ValueError(
            "WT sequence of the reference condition should have no mutations."
        )

    opt_φ = jaxopt.ProximalGradient(
        _objective_φ_preconditioned,
        prox=_prox_φ,
        value_and_grad=True,
        **block_kwargs,
    )
    opt_α = jaxopt.ProjectedGradient(
        _objective_α,
        _proj_α,
        value_and_grad=True,
        **block_kwargs,
    )

    Ps = {d: jnp.diag(1 / (1 + data_sets[d].X.sum(0).todense())) for d in data_sets}

    model = Model(
        φ={d: Latent(data_sets[d], l2reg_β=l2reg_β) for d in data_sets},
        α={d: jnp.ptp(data_sets[d].functional_scores) for d in data_sets},
        reference_condition=reference_condition,
    )

    kwargs = dict(l2reg_α=l2reg_α, l2reg_β=l2reg_β)

    filter_spec = Model(φ=True, α=False, reference_condition=reference_condition)

    obj_traj = [_objective(model, data_sets, fusionreg=fusionreg, **kwargs)]

    for iter in range(maxiter):
        try:
            print(f"iteration {iter + 1}")

            model_φ, model_α = eqx.partition(model, filter_spec)
            model_α, state_α = opt_α.run(model_α, None, model_φ, data_sets, **kwargs)
            model = eqx.combine(model_φ, model_α)
            obj_traj.append(_objective(model, data_sets, fusionreg=fusionreg, **kwargs))
            print(f"  α block:")
            print(f"    iterations={state_α.iter_num}")
            print(f"    error={state_α.error:.2e}")
            print(f"    α={jax.tree_map(lambda x: round(float(x), 2), model.α)}")

            model_φ, model_α = eqx.partition(model, filter_spec)
            model_φ, state_φ = opt_φ.run(
                model_φ,
                dict(fusionreg=fusionreg, Ps=Ps),
                model_α,
                data_sets,
                Ps,
                **kwargs,
            )
            model = eqx.combine(model_φ, model_α)
            obj_traj.append(
                _objective(model, data_sets, fusionreg=fusionreg, **kwargs)
            )
            print(f"  φ block:")
            print(f"    iterations={state_φ.iter_num}")
            print(f"    error={state_φ.error:.2e}")
            for d in data_sets:
                if d != reference_condition:
                    shifts = model.φ[d].β - model.φ[reference_condition].β
                    print(
                        f"    {d} shift sparsity={(shifts == 0).sum() / len(shifts):.2%}"
                    )

            Δobj = (obj_traj[-3] - obj_traj[-1]) / max(
                jnp.abs(obj_traj[-3]), jnp.abs(obj_traj[-1]), 1
            )
            print(f"  {Δobj=:.2e}")

            if Δobj < obj_tol:
                break
        except KeyboardInterrupt:
            print(f"Interrupted at iteration {iter + 1}")
            break

    return model, obj_traj


# The following private functions are used internally by the fit function


def _objective(model, data_sets, fusionreg=0.0, l2reg_α=0.0, l2reg_β=0.0):
    return _objective_smooth(
        model, data_sets, l2reg_α=l2reg_α, l2reg_β=l2reg_β
    ) + fusionreg * sum(
        jnp.abs(model.φ[d].β - model.φ[model.reference_condition].β).sum()
        for d in model.φ
        if d != model.reference_condition
    )


@jax.jit
def _objective_smooth(model, data_sets, l2reg_α=0.0, l2reg_β=0.0):
    n = sum(data_set.X.shape[0] for data_set in data_sets.values())
    loss = sum(model.loss(data_sets).values())
    ridge_α = sum((model.α[d] ** 2).sum() for d in model.φ)
    # NOTE: ridge wrt mean
    ridge_β = sum(((model.φ[d].β - model.φ[d].β.mean()) ** 2).sum() for d in model.φ)
    return loss / n + l2reg_α * ridge_α + l2reg_β * ridge_β


@jax.jit
@jax.value_and_grad
def _objective_φ(model_φ, model_α, data_sets, l2reg_α=0.0, l2reg_β=0.0):
    model = eqx.combine(model_φ, model_α)
    return _objective_smooth(model, data_sets, l2reg_α=l2reg_α, l2reg_β=l2reg_β)


@jax.jit
def _objective_φ_preconditioned(
    model_φ, model_α, data_sets, Ps, l2reg_α=0.0, l2reg_β=0.0
):
    value, grad = _objective_φ(
        model_φ, model_α, data_sets, l2reg_α=l2reg_α, l2reg_β=l2reg_β
    )
    for d in data_sets:
        gradβ = grad.φ[d].β
        # apply preconditioner
        P = Ps[d]
        gradβ = P @ gradβ
        grad = eqx.tree_at(lambda model_: model_.φ[d].β, grad, gradβ)
    return value, grad


@jax.jit
@jax.value_and_grad
def _objective_α(model_α, model_φ, data_sets, l2reg_α=0.0, l2reg_β=0.0):
    model = eqx.combine(model_α, model_φ)
    return _objective_smooth(model, data_sets, l2reg_α=l2reg_α, l2reg_β=l2reg_β)


@jax.jit
def _prox_φ(model, hyperparameters, scaling=1.0):
    # NOTE: for preconditioned proximal gradient, we need to use the preconditioning matrix P in the prox
    fusionreg = hyperparameters["fusionreg"]
    Ps = hyperparameters["Ps"]
    β_ref = model.φ[model.reference_condition].β
    for d in model.φ:
        if d != model.reference_condition:
            β = model.φ[d].β
            P = Ps[d]
            β_prox = β_ref + jaxopt.prox.prox_lasso(
                β - β_ref, fusionreg * jnp.diag(P), scaling
            )
            model = eqx.tree_at(lambda model_: model_.φ[d].β, model, β_prox)
    return model


@jax.jit
def _proj_α(model, hyperparameters=None):
    # box constrain alpha to the non-negative orthant
    for d in model.φ:
        model = eqx.tree_at(
            lambda model: model.α[d], model, jnp.clip(model.α[d], 0.0, jnp.inf)
        )
    return model
