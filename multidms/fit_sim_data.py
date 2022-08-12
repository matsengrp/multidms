#!/usr/bin/env python


# python native
import pickle
from timeit import default_timer as timer
import os

# dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental import sparse
import jaxopt

# local
from utils import *
from model import *

simulated_dataset = pd.read_csv("../results/simulated_dataset_v1.csv")
simulated_dataset_lib1 = simulated_dataset.query("library == 'lib_1'").copy()
simulated_dataset_lib1.aa_substitutions.fillna("", inplace=True)
#simulated_dataset_lib1 = simulated_dataset_lib1.sample(n=20000, random_state=23)

homologs = json.load(open("../results/homolog_aa_seqs.json", "r"))
homologs["reference"] = homologs['1']
homologs["H2"] = homologs['2']
del homologs['1']
del homologs['2']

(X, y), df, all_subs = create_homolog_modeling_data(
                            simulated_dataset_lib1, 
                            homologs, 
                            "homolog", 
                            "aa_substitutions",
                            "observed_phenotype"
                        )

# number of columns in the sparse arrays are the number of beta/shift params to fit
params = initialize_model_params(homologs, n_beta_shift_params = X["reference"].shape[1])

print(f"\nParameter Shapes")
print(f"----------------")
for key, value in params.items():
    if key == "α":
        for key_a, value_a in value.items():
            print(f"Parameter {key_a} has shape: {value_a.shape}")        
    else:
        print(f"Parameter {key} has shape: {value.shape}")

print(f"\nPre-Optimization")
print(f"----------------")
print(f"cost = {cost_smooth(params, (X, y)):.2e}")

tol = 1e-6
maxiter = 100
solver = jaxopt.GradientDescent(cost_smooth, tol=tol, maxiter=maxiter)

start = timer()
params, state = solver.run(params, data=(X, y))
end = timer()

print(f"\nPost-Optimization")
print(f"-----------------")
print(f"Full model optimization: {state.iter_num} iterations")
print(f"error = {state.error:.2e}")
print(f"cost = {cost_smooth(params, (X, y)):.2e}")
print(f"Wall time for fit: {end - start}")

for param in ["β", "S_reference", "S_H2"]:
    # print(params[param].)
    print(f"\nFit {param} distribution\n===============")
    arr = np.array(params[param])
    mean = np.mean(arr)
    median = np.median(arr)
     
    # measures of dispersion
    min = np.amin(arr)
    max = np.amax(arr)
    range = np.ptp(arr)
    variance = np.var(arr)
    sd = np.std(arr)
     
    print("Descriptive analysis")
    print("Measures of Central Tendency")
    print(f"Mean = {mean:.2e}")
    print(f"Median = {median:.2e}")
    print("Measures of Dispersion")
    print(f"Minimum = {min:.2e}")
    print(f"Maximum = {max:.2e}")
    print(f"Range = {range:.2e}")
    print(f"Variance = {variance:.2e}")
    print(f"Standard Deviation = {sd:.2e}")

print(f"\nFit Sigmoid Parameters, α\n================")
for param, value in params['α'].items():
    print(f"{param}: {value[0]:.2e}") 

df["latent_predicted"] = onp.nan
df["observed_predicted"] = onp.nan

print(f"\nRunning Predictions")
print(f"-------------------")
for homolog, hdf in df.groupby("homolog"):

    h_params = {"β":params["β"], "S":params[f"S_{homolog}"], "C_ref":params["C_ref"]}
    z_h = ϕ(h_params, X[homolog])
    df.loc[hdf.index, "latent_predicted"] = z_h
    y_h_pred = g(params["α"], z_h)
    df.loc[hdf.index, "observed_predicted"] = y_h_pred

print(f"Done")


print(f"\nPlotting")
print(f"--------")
fig, ax = plt.subplots(1, 3, figsize=[10, 8])
sns.scatterplot(data=df, x="observed_phenotype", y="observed_predicted",
                hue="n_aa_substitutions",
                alpha=0.2, palette="deep", ax=ax[0],
                legend=False)

lb = df[["observed_phenotype", "observed_predicted"]].min().min()
ub = df[["observed_phenotype", "observed_predicted"]].max().max()

ax[0].plot([lb, ub], [lb, ub], "k--", lw=1)
r = pearsonr(df.observed_phenotype, df.observed_predicted)[0]
ax[0].annotate(f"$r = {r:.2f}$", (.5, .9), xycoords="axes fraction", fontsize=12)

sns.scatterplot(data=df, x="latent_phenotype", y="latent_predicted",
                hue="n_aa_substitutions",
                alpha=0.2, palette="deep", ax=ax[1],
                legend=False)

sns.scatterplot(data=df, x="latent_predicted", y="observed_predicted",
                hue="n_aa_substitutions",
                alpha=0.2, palette="deep",
                legend=False, ax=ax[2])

plt.tight_layout()
fig.savefig("../_ignore/eval-scatter.png")
print(f"Done")


# ϕ_grid = onp.linspace(1.1 * df.latent_predicted.min(), 1.1 * df.latent_predicted.max())
# plt.plot(ϕ_grid, g(ϕ_grid, params["α"]), "k")
# plt.axhline(0, color="k", ls="--", lw=1)
# plt.axvline(0, color="k", ls="--", lw=1)

fig, ax = plt.subplots(3, 1, figsize=(55, 15))
for i, param in enumerate(["β", "S_reference", "S_H2"]):
    rows = []
    for mutation, p in zip(all_subs, params[param]):
        wt = mutation[0]
        mutant = mutation[-1]
        site = int(mutation[1:-1])
        rows.append([site, wt, mutant, float(p)])

    mutation_effects = pd.DataFrame(
        rows,
        columns=("site", "wildtype", "mutant", param)
    ).pivot(
        index="mutant",
        columns="site", values=param
    )

    sns.heatmap(mutation_effects, mask=mutation_effects.isnull(),
                linewidths=1,
                cmap="coolwarm_r", center=0,
                # vmin=-1, vmax=1,
                cbar_kws={"label": param},
                ax=ax[i])

fig.savefig(f"../_ignore/heatmaps.png")
