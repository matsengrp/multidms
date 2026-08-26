"""Pick one real spike variant and walk it through BOTH frameworks end-to-end.

Prints every intermediate number so the worked example in the issue uses real data.
Reuses the same parameterization as compute_equivalence.py.
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse
import multidms

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

REFERENCE = "Omicron_BA1"
CHOSEN_LASSO = 4e-5
REPLICATE = "rep-2"
REPLICATE_NUM = 2
DELTA = 1.0
THETA_SCALE = 7.4
PHI_WT_READ = {"Delta": -0.4, "Omicron_BA1": -0.4, "Omicron_BA2": -0.4}
THETA_BIAS_CAL = -THETA_SCALE / (1 + np.exp(-(-0.4)))  # = -theta_scale*sigmoid(phi_wt_ref); D=0

EXAMPLE_CONDITION = "Omicron_BA1"   # the REFERENCE homolog: WT is all-zero, so no
                                    # homolog "bundle" -- phi is just the variant's own
                                    # substitutions and beta_d = beta (shift = 0).


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def huber(r, delta=DELTA):
    a = abs(r)
    return 0.5 * a * a if a <= delta else delta * (a - 0.5 * delta)


fs = pd.read_csv(os.path.join(DATA, "training_functional_scores.csv"))
fs = fs[fs["replicate"] == REPLICATE_NUM].copy()
fs["aa_substitutions"] = fs["aa_substitutions"].fillna("")
variants = fs.groupby(["condition", "aa_substitutions"], as_index=False)["func_score"].mean()

mdata = multidms.Data(variants, reference=REFERENCE,
                      alphabet=multidms.AAS_WITHSTOP_WITHGAP, assert_site_integrity=False)
muts = list(mdata.mutations)
idx = {m: i for i, m in enumerate(muts)}
conditions = list(mdata.conditions)

md_ = pd.read_csv(os.path.join(DATA, "mutations_df.csv"))
md_ = md_[(md_["dataset_name"] == REPLICATE)
          & np.isclose(md_["scale_coeff_lasso_shift"].astype(float), CHOSEN_LASSO)].set_index("mutation")
beta = np.zeros(len(muts))
shift = {c: np.zeros(len(muts)) for c in conditions}
for m in muts:
    if m in md_.index:
        beta[idx[m]] = md_.at[m, "beta"]
        shift["Delta"][idx[m]] = md_.at[m, "shift_Delta"]
        shift["Omicron_BA2"][idx[m]] = md_.at[m, "shift_Omicron_BA2"]
beta_d = {c: beta + shift[c] for c in conditions}


def cond_arrays(c):
    X = mdata.arrays["X"][c]
    Xs = scipy.sparse.csr_array((np.asarray(X.data),
        (np.asarray(X.indices[:, 0]), np.asarray(X.indices[:, 1]))), shape=X.shape)
    x_wt = np.asarray(Xs[[0], :].todense()).ravel()
    return Xs, x_wt

_, x_wt_c = cond_arrays(EXAMPLE_CONDITION)
beta0_c = PHI_WT_READ[EXAMPLE_CONDITION] - float(x_wt_c @ beta_d[EXAMPLE_CONDITION])

# choose a readable variant: exactly 2 substitutions, both present in mutations_df,
# moderate observed score, in the example condition
cand = variants[variants["condition"] == EXAMPLE_CONDITION].copy()
cand["subs"] = cand["aa_substitutions"].str.split()
cand["nsub"] = cand["subs"].apply(lambda s: len([x for x in s if x]))
cand = cand[cand["nsub"] == 2]
cand = cand[cand["subs"].apply(lambda s: all(m in idx and m in md_.index for m in s))]
cand = cand[(cand["func_score"] > -1.5) & (cand["func_score"] < 0.5)]
row = cand.sort_values("func_score").iloc[len(cand) // 2]   # a median-ish one
subs = [m for m in row["subs"] if m]
y = float(row["func_score"])

print(f"condition   = {EXAMPLE_CONDITION}  (reference = {REFERENCE})")
print(f"variant     = {row['aa_substitutions']!r}")
print(f"observed y  = {y:.4f}")
print(f"beta0_{EXAMPLE_CONDITION} = {beta0_c:.4f}   (from read phi_wt={PHI_WT_READ[EXAMPLE_CONDITION]})")
print(f"theta_scale = {THETA_SCALE}   theta_bias(calibrated) = {THETA_BIAS_CAL:.4f}")
tot = beta0_c
for m in subs:
    b = beta[idx[m]]; d = shift[EXAMPLE_CONDITION][idx[m]]
    print(f"  mut {m:10s}  beta={b:+.4f}  shift_{EXAMPLE_CONDITION}={d:+.4f}  beta_d={b+d:+.4f}")
    tot += b + d
phi = tot
phi_wt = beta0_c + float(x_wt_c @ beta_d[EXAMPLE_CONDITION])

# cross-check phi via the full reference-frame encoding (guards against a hidden bundle:
# for the reference condition this must equal the manual sum over the variant's own subs)
bmap = mdata.binarymaps[EXAMPLE_CONDITION]
bmap_subs = list(bmap.all_subs)
bd = beta_d[EXAMPLE_CONDITION]
bd_bmap = np.array([bd[idx[m]] for m in bmap_subs])
bin_vec = np.asarray(bmap.sub_str_to_binary(row["aa_substitutions"])).ravel()
phi_enc = beta0_c + float(bin_vec @ bd_bmap)
n_enc_muts = int(bin_vec.sum())
print(f"phi (manual sum over variant subs) = {phi:.4f}")
print(f"phi (full ref-frame encoding)      = {phi_enc:.4f}   "
      f"encoding nonzero muts = {n_enc_muts}  match={np.isclose(phi, phi_enc, atol=1e-6)}")
print(f"phi (variant latent)        = {phi:.4f}   sigma(phi)   = {sigmoid(phi):.4f}")
print(f"phi_wt (condition WT latent)= {phi_wt:.4f}   sigma(phi_wt)= {sigmoid(phi_wt):.4f}")

pred_040 = THETA_SCALE * sigmoid(phi) + THETA_BIAS_CAL
pred_main = THETA_SCALE * (sigmoid(phi) - sigmoid(phi_wt))
D_d = THETA_BIAS_CAL + THETA_SCALE * sigmoid(phi_wt)
print(f"\nD_d = theta_bias + theta_scale*sigma(phi_wt) = {D_d:+.6f}")
print(f"0.4.0 predicted score = theta_scale*sigma(phi) + theta_bias = {pred_040:.4f}")
print(f"main  predicted score = alpha*(sigma(phi) - sigma(phi_wt))  = {pred_main:.4f}")
print(f"residual 0.4.0 = {pred_040 - y:+.4f}   Huber = {huber(pred_040 - y):.6f}")
print(f"residual main  = {pred_main - y:+.4f}   Huber = {huber(pred_main - y):.6f}")

# ------------------------------------------------------------------
# Does the reference-frame encoding (used by the correlation code) include the
# homolog "bundle"?  For each condition print the WT's ref-frame mutation count
# (= bundle size) and, for a sample variant, the encoding count vs its own subs.
print("\n==== bundle check: encoding used by the correlation code (Xv @ beta_d) ====")
for c in conditions:
    Xc = mdata.arrays["X"][c]
    Xs = scipy.sparse.csr_array((np.asarray(Xc.data),
        (np.asarray(Xc.indices[:, 0]), np.asarray(Xc.indices[:, 1]))), shape=Xc.shape)
    rownnz = np.asarray((Xs != 0).sum(axis=1)).ravel()   # ref-frame mutations per row
    bundle = int(rownnz[0])                              # row 0 = homolog WT
    var_nnz = rownnz[1:]                                 # variant rows
    print(f"  {c:16s} WT row (bundle)={bundle:4d} | variant rows ref-frame muts: "
          f"min={var_nnz.min()} median={int(np.median(var_nnz))} max={var_nnz.max()}")
