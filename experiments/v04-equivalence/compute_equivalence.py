"""Demonstrate the 0.4.0 -> main (PR #164) transformation preserves the Huber loss.

Parameters for the sigmoid global-epistasis shape are read approximately from
``ge_fits.png`` (no fitted pickle used); the per-mutation effects/shifts and the
observed functional scores come from the committed spike-analysis CSVs.

The two forward models share one latent phenotype phi_d(v); the map is
    beta_d  = beta + shift_d          (per-condition absolute effect vector)
    beta0_d = phi_wt_d - x_wt_d . beta_d   (per-condition intercept, from the read WT latent)
    alpha   = theta_scale             (shared output scale)
0.4.0 predicts   theta_scale * sigma(phi) + theta_bias        (raw; WT not subtracted in loss)
main  predicts   alpha * (sigma(phi) - sigma(phi_wt_d))       (WT subtracted structurally)
They differ per condition only by  D_d = theta_bias + theta_scale * sigma(phi_wt_d);
the losses coincide as D_d -> 0.
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

REFERENCE = "Omicron_BA1"
CHOSEN_LASSO = 4e-5
REPLICATE = "rep-2"          # committed replicate label in mutations_df.csv
REPLICATE_NUM = 2           # matching value in training_functional_scores.csv

# ----- GE-shape parameters read approximately from ge_fits.png -----
THETA_SCALE = 7.4           # sigmoid range (midpoint read; upper asymptote off-screen)
THETA_BIAS_RAW = -3.4       # lower asymptote
# WT vertical lines cluster near phi = -0.4 for all conditions (replicate 2 panel)
PHI_WT_READ = {"Delta": -0.4, "Omicron_BA1": -0.4, "Omicron_BA2": -0.4}
DELTA = 1.0                 # Huber delta, matches 0.4.0 and main defaults


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def huber(resid, delta=DELTA):
    a = np.abs(resid)
    return np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))


def main():
    # -------- fetch committed spike-analysis CSVs (pinned commit) if missing --------
    import urllib.request
    os.makedirs(DATA, exist_ok=True)
    base = ("https://raw.githubusercontent.com/matsengrp/SARS-CoV-2_spike_multidms/"
            "6c98b7b607d7387b508cdaa192d659ee9fca7367/results/spike_analysis")
    for f in ["mutations_df.csv", "training_functional_scores.csv"]:
        p = os.path.join(DATA, f)
        if not os.path.exists(p):
            print("downloading", f)
            urllib.request.urlretrieve(f"{base}/{f}", p)

    # -------- observed functional scores (one replicate) --------
    fs = pd.read_csv(os.path.join(DATA, "training_functional_scores.csv"))
    fs = fs[fs["replicate"] == REPLICATE_NUM].copy()
    fs["aa_substitutions"] = fs["aa_substitutions"].fillna("")
    # Collapse identical variants by mean (matches the fit's
    # collapse_identical_variants="mean"), which averages repeated measurements.
    variants = (
        fs.groupby(["condition", "aa_substitutions"], as_index=False)["func_score"].mean()
    )
    print(f"variants (replicate {REPLICATE_NUM}): raw={len(fs)}  collapsed={len(variants)}  "
          f"conditions: {sorted(variants['condition'].unique())}")

    mdata = multidms.Data(
        variants,
        reference=REFERENCE,
        alphabet=multidms.AAS_WITHSTOP_WITHGAP,
        assert_site_integrity=False,
    )
    muts = list(mdata.mutations)
    idx = {m: i for i, m in enumerate(muts)}
    conditions = list(mdata.conditions)
    print(f"n_mutations: {len(muts)}  conditions: {conditions}  reference: {mdata.reference}")

    # -------- per-mutation effects / shifts (chosen lasso, chosen replicate) --------
    md = pd.read_csv(os.path.join(DATA, "mutations_df.csv"))
    md = md[(md["dataset_name"] == REPLICATE)
            & np.isclose(md["scale_coeff_lasso_shift"].astype(float), CHOSEN_LASSO)].copy()
    md = md.set_index("mutation")
    beta = np.zeros(len(muts))
    shift = {c: np.zeros(len(muts)) for c in conditions}
    n_hit = 0
    for m in muts:
        if m in md.index:
            n_hit += 1
            beta[idx[m]] = md.at[m, "beta"]
            shift["Delta"][idx[m]] = md.at[m, "shift_Delta"]
            shift["Omicron_BA2"][idx[m]] = md.at[m, "shift_Omicron_BA2"]
    print(f"mutations matched to mutations_df: {n_hit}/{len(muts)}")
    beta_d = {c: beta + shift[c] for c in conditions}

    # -------- per-condition sparse encodings (WT is row 0) --------
    def cond_arrays(c):
        X = mdata.arrays["X"][c]                      # BCOO, WT at row 0
        Xs = scipy.sparse.csr_array(
            (np.asarray(X.data), (np.asarray(X.indices[:, 0]), np.asarray(X.indices[:, 1]))),
            shape=X.shape,
        )
        x_wt = np.asarray(Xs[[0], :].todense()).ravel()
        Xv = Xs[1:]                                   # drop WT row
        y = np.asarray(mdata.arrays["y"][c])[1:]
        return Xv, x_wt, y

    # intercept from the read WT latent:  phi_wt = beta0 + x_wt . beta_d
    beta0 = {}
    for c in conditions:
        _, x_wt, _ = cond_arrays(c)
        beta0[c] = PHI_WT_READ[c] - float(x_wt @ beta_d[c])

    theta_bias_cal = -THETA_SCALE * sigmoid(PHI_WT_READ[REFERENCE])  # enforce D_ref = 0

    for label, theta_bias in [("raw read", THETA_BIAS_RAW), ("calibrated (D_ref=0)", theta_bias_cal)]:
        print(f"\n================  theta_bias = {theta_bias:+.4f}  [{label}]  "
              f"theta_scale = {THETA_SCALE}  ================")
        print(f"{'condition':16s} {'D_d':>9s} {'loss_0.4.0':>12s} {'loss_main':>12s} {'|gap|':>10s}")
        rows = []
        for c in conditions:
            Xv, x_wt, y = cond_arrays(c)
            phi = beta0[c] + np.asarray(Xv @ beta_d[c]).ravel()
            phi_wt = beta0[c] + float(x_wt @ beta_d[c])
            D_d = theta_bias + THETA_SCALE * sigmoid(phi_wt)
            pred_040 = THETA_SCALE * sigmoid(phi) + theta_bias
            pred_main = THETA_SCALE * (sigmoid(phi) - sigmoid(phi_wt))
            loss_040 = float(huber(pred_040 - y).mean())
            loss_main = float(huber(pred_main - y).mean())
            print(f"{c:16s} {D_d:+9.4f} {loss_040:12.6f} {loss_main:12.6f} {abs(loss_040-loss_main):10.2e}")
            rows.append((c, D_d, loss_040, loss_main))
        tot040 = sum(r[2] for r in rows)
        totmain = sum(r[3] for r in rows)
        print(f"{'TOTAL (sum)':16s} {'':>9s} {tot040:12.6f} {totmain:12.6f} {abs(tot040-totmain):10.2e}")

    # ---------- observed-vs-predicted correlation (validates the parameterization) ----------
    # Pearson r within a condition is invariant to affine transforms of the prediction,
    # so it is identical for the 0.4.0 and main forms and (nearly) independent of the GE
    # shape params -- it validates the latent phi (i.e. beta/shift/beta0) reconstruction.
    print("\n---- observed vs predicted functional score: Pearson r per condition ----")
    print(f"(target from func_score_corr.png: Delta ~0.80, {REFERENCE} ~0.86-0.90, Omicron_BA2 ~0.86-0.88)")
    for c in conditions:
        Xv, x_wt, y = cond_arrays(c)
        phi = beta0[c] + np.asarray(Xv @ beta_d[c]).ravel()
        phi_wt = beta0[c] + float(x_wt @ beta_d[c])
        pred_040 = THETA_SCALE * sigmoid(phi) + theta_bias_cal
        pred_main = THETA_SCALE * (sigmoid(phi) - sigmoid(phi_wt))
        m = np.isfinite(pred_040) & np.isfinite(y)
        r_040 = float(np.corrcoef(pred_040[m], y[m])[0, 1])
        r_main = float(np.corrcoef(pred_main[m], y[m])[0, 1])
        print(f"  {c:16s} r(0.4.0-form)={r_040:.3f}  r(main-form)={r_main:.3f}  (n={int(m.sum())})")

    # ---------- code-path check: main's own functional_score_loss ----------
    print("\n---- main code-path check (jaxmodels.functional_score_loss, calibrated theta_bias) ----")
    data_sets = {c: jm.Data.from_multidms(mdata, c) for c in conditions}
    model = jm.Model(
        φ={c: jm.Latent(β0=jnp.array(float(beta0[c])), β=jnp.asarray(beta_d[c]))
           for c in conditions},
        α=jnp.array(float(THETA_SCALE)),
        logθ={c: jnp.array(0.0) for c in conditions},
        reference_condition=REFERENCE,
        global_epistasis=jm.Sigmoid(),
    )
    loss = jm.functional_score_loss(model, data_sets, δ=DELTA)
    for c in conditions:
        # direct main-form loss recomputed here for the same condition
        Xv, x_wt, y = cond_arrays(c)
        phi = beta0[c] + np.asarray(Xv @ beta_d[c]).ravel()
        phi_wt = beta0[c] + float(x_wt @ beta_d[c])
        pred_main = THETA_SCALE * (sigmoid(phi) - sigmoid(phi_wt))
        direct_mean = float(huber(pred_main - y).mean())
        fscl = float(loss[c])
        n = len(y)
        print(f"  {c:16s} functional_score_loss={fscl:.6f}  direct_mean={direct_mean:.6f}  "
              f"fscl/n={fscl/n:.6f}  (matches mean: {np.isclose(fscl/n, direct_mean, atol=1e-4) or np.isclose(fscl, direct_mean, atol=1e-4)})")


if __name__ == "__main__":
    main()
