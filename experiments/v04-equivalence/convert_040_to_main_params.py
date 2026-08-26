"""Convert fitted multidms 0.4.0 parameters into the parameter set that
initializes a ``main`` (PR #164) ``jaxmodels.Model`` at the SAME point in
parameter space, and write that parameter set to CSV files.

Forward models
--------------
0.4.0 (per condition ``d``, sigmoid GE, gamma = 0)::

    phi_d(v) = beta_naught + alpha_d + X_v . (beta + shift_d)
    yhat     = theta_scale * sigmoid(phi) + theta_bias      # raw; WT not subtracted in loss

main (per condition ``d``)::

    phi_d(v) = beta0_d + X_v . beta_d
    yhat     = alpha * (sigmoid(phi) - sigmoid(phi_wt_d))   # WT subtracted structurally

Parameter map (0.4.0 -> main)
-----------------------------
    beta_d     = beta + shift_d          # per-condition absolute effect vector (ref: shift=0)
    beta0_d    = beta_naught + alpha_d   # fold the additive latent offset into the intercept
    alpha      = theta_scale             # 0.4.0 sigmoid *range* becomes main's output scale
    logtheta_d = 0                       # overdispersion; unused by functional_score_loss

``theta_bias`` has no home in main. The two per-condition Huber losses differ by
    D_d = theta_bias + theta_scale * sigmoid(phi_wt_d),
which vanishes -- making the initialization an *exact* re-expression of the fit --
when the 0.4.0 fit places each condition's wildtype at predicted score 0.

This script:
  1. loads the committed spike-analysis CSVs for one replicate (beta, shifts,
     functional scores) and the GE-shape params read from ``ge_fits.png``;
  2. converts them to the main parameter set;
  3. writes three CSVs that fully specify a ``jaxmodels.Model``;
  4. reads the CSVs back, rebuilds the ``jaxmodels.Model``, and verifies it lands
     exactly where intended (loss / predictions match a model built in-memory).
"""

import os

import numpy as np
import pandas as pd
import scipy.sparse

import multidms
import multidms.jaxmodels as jm
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUTDIR = os.path.join(HERE, "main_init_params")

# -------------------- 0.4.0 fit identifiers (spike, one replicate) --------------------
REFERENCE = "Omicron_BA1"
CHOSEN_LASSO = 4e-5
REPLICATE = "rep-2"       # committed replicate label in mutations_df.csv
REPLICATE_NUM = 2         # matching value in training_functional_scores.csv

# -------------------- GE-shape params read approximately from ge_fits.png -------------
THETA_SCALE = 7.4         # sigmoid range (midpoint read; upper asymptote off-screen)
THETA_BIAS_RAW = -3.4     # lower asymptote as eyeballed
PHI_WT_READ = {"Delta": -0.4, "Omicron_BA1": -0.4, "Omicron_BA2": -0.4}  # WT lines, rep-2 panel
DELTA = 1.0               # Huber delta, matches 0.4.0 and main defaults


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ======================================================================================
# 1. Load the 0.4.0 parameters for one spike replicate
# ======================================================================================
def load_spike_040_params():
    """Return (mdata, conditions, muts, beta_d, x_wt, phi_wt) for the replicate.

    ``beta_d[c]`` = beta + shift_c is the per-condition absolute effect vector in
    the mutation order of ``mdata`` (the order ``jaxmodels`` expects for ``Latent.β``).
    ``x_wt[c]`` is the reference-frame encoding of condition ``c``'s wildtype
    (the homolog "bundle"), and ``phi_wt[c]`` is that wildtype's latent read off
    ``ge_fits.png``.
    """
    import urllib.request

    os.makedirs(DATA, exist_ok=True)
    base = (
        "https://raw.githubusercontent.com/matsengrp/SARS-CoV-2_spike_multidms/"
        "6c98b7b607d7387b508cdaa192d659ee9fca7367/results/spike_analysis"
    )
    for f in ["mutations_df.csv", "training_functional_scores.csv"]:
        p = os.path.join(DATA, f)
        if not os.path.exists(p):
            print("downloading", f)
            urllib.request.urlretrieve(f"{base}/{f}", p)

    fs = pd.read_csv(os.path.join(DATA, "training_functional_scores.csv"))
    fs = fs[fs["replicate"] == REPLICATE_NUM].copy()
    fs["aa_substitutions"] = fs["aa_substitutions"].fillna("")
    # Collapse identical variants by mean, matching the fit's
    # collapse_identical_variants="mean".
    variants = fs.groupby(
        ["condition", "aa_substitutions"], as_index=False
    )["func_score"].mean()

    mdata = multidms.Data(
        variants,
        reference=REFERENCE,
        alphabet=multidms.AAS_WITHSTOP_WITHGAP,
        assert_site_integrity=False,
    )
    muts = list(mdata.mutations)
    idx = {m: i for i, m in enumerate(muts)}
    conditions = list(mdata.conditions)

    md = pd.read_csv(os.path.join(DATA, "mutations_df.csv"))
    md = md[
        (md["dataset_name"] == REPLICATE)
        & np.isclose(md["scale_coeff_lasso_shift"].astype(float), CHOSEN_LASSO)
    ].set_index("mutation")

    beta = np.zeros(len(muts))
    shift = {c: np.zeros(len(muts)) for c in conditions}
    for m in muts:
        if m in md.index:
            beta[idx[m]] = md.at[m, "beta"]
            shift["Delta"][idx[m]] = md.at[m, "shift_Delta"]
            shift["Omicron_BA2"][idx[m]] = md.at[m, "shift_Omicron_BA2"]
    beta_d = {c: beta + shift[c] for c in conditions}  # ref shift == 0

    # reference-frame WT encoding (row 0 of each condition's X) = homolog bundle
    x_wt = {}
    for c in conditions:
        X = mdata.arrays["X"][c]
        Xs = scipy.sparse.csr_array(
            (
                np.asarray(X.data),
                (np.asarray(X.indices[:, 0]), np.asarray(X.indices[:, 1])),
            ),
            shape=X.shape,
        )
        x_wt[c] = np.asarray(Xs[[0], :].todense()).ravel()

    phi_wt = dict(PHI_WT_READ)
    return mdata, conditions, muts, beta_d, x_wt, phi_wt


# ======================================================================================
# 2. Convert 0.4.0 -> main parameter set
# ======================================================================================
def convert_040_to_main(conditions, beta_d, x_wt, phi_wt, theta_scale,
                         theta_bias_raw, reference):
    """Map fitted 0.4.0 parameters to the ``main`` parameter set.

    Args:
        conditions: ordered list of condition names.
        beta_d: dict condition -> absolute effect vector (beta + shift_d).
        x_wt: dict condition -> reference-frame WT encoding.
        phi_wt: dict condition -> wildtype latent (beta_naught + alpha_d + x_wt.beta_d).
        theta_scale: 0.4.0 sigmoid range.
        theta_bias_raw: 0.4.0 sigmoid lower asymptote (as read).
        reference: reference condition name.

    Returns:
        dict with keys ``beta0`` (per condition), ``beta`` (per condition vectors),
        ``alpha`` (scalar), ``logtheta`` (per condition), ``reference``,
        and diagnostics ``phi_wt``, ``D_d_raw`` (loss offset under the raw bias),
        ``theta_bias_calibrated`` (the bias that makes D_d == 0).
    """
    # main's per-condition intercept folds beta_naught + alpha_d together; we do not
    # observe them separately, but phi_wt_d = beta0_d + x_wt_d . beta_d pins beta0_d.
    beta0 = {c: float(phi_wt[c] - x_wt[c] @ beta_d[c]) for c in conditions}
    alpha = float(theta_scale)
    logtheta = {c: 0.0 for c in conditions}

    # calibrated bias that enforces exact equivalence (D_d == 0) at the reference WT
    theta_bias_cal = -theta_scale * sigmoid(phi_wt[reference])
    D_d_raw = {c: theta_bias_raw + theta_scale * sigmoid(phi_wt[c]) for c in conditions}

    return {
        "beta0": beta0,
        "beta": {c: np.asarray(beta_d[c]) for c in conditions},
        "alpha": alpha,
        "logtheta": logtheta,
        "reference": reference,
        "phi_wt": dict(phi_wt),
        "D_d_raw": D_d_raw,
        "theta_bias_calibrated": float(theta_bias_cal),
    }


# ======================================================================================
# 3. Write the main parameter set to CSVs
# ======================================================================================
def write_csvs(params, muts, conditions, outdir=OUTDIR):
    """Write three CSVs that fully specify the main jaxmodels.Model.

    - ``main_params_beta.csv``   : long table (condition, mutation, beta) = phi[c].beta
    - ``main_params_latent.csv`` : per condition (beta0, logtheta, is_reference, phi_wt, D_d_raw)
    - ``main_params_global.csv`` : key/value scalars & structural choices (alpha, reference, GE, ...)
    """
    os.makedirs(outdir, exist_ok=True)

    beta_rows = []
    for c in conditions:
        b = params["beta"][c]
        for m, v in zip(muts, b):
            beta_rows.append({"condition": c, "mutation": m, "beta": float(v)})
    beta_df = pd.DataFrame(beta_rows)
    beta_path = os.path.join(outdir, "main_params_beta.csv")
    beta_df.to_csv(beta_path, index=False)

    latent_df = pd.DataFrame(
        [
            {
                "condition": c,
                "beta0": params["beta0"][c],
                "logtheta": params["logtheta"][c],
                "is_reference": c == params["reference"],
                "phi_wt": params["phi_wt"][c],
                "D_d_raw_bias": params["D_d_raw"][c],
            }
            for c in conditions
        ]
    )
    latent_path = os.path.join(outdir, "main_params_latent.csv")
    latent_df.to_csv(latent_path, index=False)

    global_df = pd.DataFrame(
        [
            {"parameter": "alpha", "value": params["alpha"]},
            {"parameter": "reference_condition", "value": params["reference"]},
            {"parameter": "global_epistasis", "value": "Sigmoid"},
            {"parameter": "output_activation", "value": "IdentityOutput"},
            {"parameter": "huber_delta", "value": DELTA},
            {"parameter": "theta_scale_source", "value": THETA_SCALE},
            {"parameter": "theta_bias_calibrated", "value": params["theta_bias_calibrated"]},
        ]
    )
    global_path = os.path.join(outdir, "main_params_global.csv")
    global_df.to_csv(global_path, index=False)

    return beta_path, latent_path, global_path


# ======================================================================================
# 4. Rebuild the model from the CSVs and verify it lands where intended
# ======================================================================================
def build_model_from_csvs(mdata, outdir=OUTDIR):
    """Reconstruct a jaxmodels.Model purely from the written CSVs."""
    muts = list(mdata.mutations)
    idx = {m: i for i, m in enumerate(muts)}

    beta_df = pd.read_csv(os.path.join(outdir, "main_params_beta.csv"))
    latent_df = pd.read_csv(os.path.join(outdir, "main_params_latent.csv"))
    global_df = pd.read_csv(os.path.join(outdir, "main_params_global.csv")).set_index(
        "parameter"
    )["value"]

    alpha = float(global_df["alpha"])
    reference = str(global_df["reference_condition"])
    conditions = list(latent_df["condition"])

    φ = {}
    logθ = {}
    for _, r in latent_df.iterrows():
        c = r["condition"]
        b = np.zeros(len(muts))
        sub = beta_df[beta_df["condition"] == c]
        for _, br in sub.iterrows():
            b[idx[br["mutation"]]] = br["beta"]  # align to mdata mutation order
        φ[c] = jm.Latent(β0=jnp.array(float(r["beta0"])), β=jnp.asarray(b))
        logθ[c] = jnp.array(float(r["logtheta"]))

    model = jm.Model(
        φ=φ,
        α=jnp.array(alpha),
        logθ=logθ,
        reference_condition=reference,
        global_epistasis=jm.Sigmoid(),
    )
    return model, conditions


def main():
    print(f"Loading 0.4.0 spike parameters (replicate {REPLICATE!r}, lasso {CHOSEN_LASSO}) ...")
    mdata, conditions, muts, beta_d, x_wt, phi_wt = load_spike_040_params()
    print(f"  conditions={conditions}  reference={mdata.reference}  n_mutations={len(muts)}")

    params = convert_040_to_main(
        conditions, beta_d, x_wt, phi_wt,
        theta_scale=THETA_SCALE, theta_bias_raw=THETA_BIAS_RAW, reference=REFERENCE,
    )

    print("\nmain parameter set:")
    print(f"  alpha (= theta_scale)          = {params['alpha']}")
    print(f"  theta_bias (calibrated, D=0)   = {params['theta_bias_calibrated']:+.4f}")
    for c in conditions:
        print(
            f"  {c:16s} beta0={params['beta0'][c]:+.4f}  logtheta={params['logtheta'][c]:.1f}  "
            f"phi_wt={params['phi_wt'][c]:+.2f}  D_d(raw bias={THETA_BIAS_RAW})={params['D_d_raw'][c]:+.4f}"
        )

    beta_path, latent_path, global_path = write_csvs(params, muts, conditions)
    print("\nwrote:")
    for p in (beta_path, latent_path, global_path):
        print(f"  {p}")

    # ---- verify: CSV round-trip builds a model identical to the in-memory one ----
    print("\nVerifying the CSVs initialize the model at the intended point ...")
    model_csv, _ = build_model_from_csvs(mdata)
    model_mem = jm.Model(
        φ={c: jm.Latent(β0=jnp.array(float(params["beta0"][c])),
                        β=jnp.asarray(params["beta"][c])) for c in conditions},
        α=jnp.array(params["alpha"]),
        logθ={c: jnp.array(float(params["logtheta"][c])) for c in conditions},
        reference_condition=REFERENCE,
        global_epistasis=jm.Sigmoid(),
    )
    data_sets = {c: jm.Data.from_multidms(mdata, c) for c in conditions}

    loss_csv = jm.functional_score_loss(model_csv, data_sets, δ=DELTA)
    loss_mem = jm.functional_score_loss(model_mem, data_sets, δ=DELTA)
    pred_csv = model_csv.predict_score(data_sets)

    print(f"{'condition':16s} {'β0(csv)':>10s} {'loss(csv)':>12s} {'loss(mem)':>12s} {'|gap|':>10s}")
    all_ok = True
    for c in conditions:
        b0 = float(model_csv.φ[c].β0)
        lc, lm = float(loss_csv[c]), float(loss_mem[c])
        gap = abs(lc - lm)
        all_ok &= gap < 1e-9 and np.allclose(
            np.asarray(model_csv.φ[c].β), np.asarray(params["beta"][c])
        )
        print(f"{c:16s} {b0:+10.4f} {lc:12.6f} {lm:12.6f} {gap:10.2e}")

    # sanity: predicted vs observed correlation (validates the parameter point)
    print("\nobserved vs predicted Pearson r (should match func_score_corr.png ~0.8-0.9):")
    for c in conditions:
        y = np.asarray(data_sets[c].functional_scores)
        yhat = np.asarray(pred_csv[c])
        m = np.isfinite(y) & np.isfinite(yhat)
        r = float(np.corrcoef(yhat[m], y[m])[0, 1])
        print(f"  {c:16s} r={r:.3f}  (n={int(m.sum())})")

    print(f"\nCSV round-trip {'OK -- model initialized at the intended point.' if all_ok else 'MISMATCH!'}")


if __name__ == "__main__":
    main()
