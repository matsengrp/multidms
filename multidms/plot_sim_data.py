#!/usr/bin/env python


# python native
import pickle
from timeit import default_timer as timer
import os

# dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

ps = "ref_to_full"

(params, (X, y), df, simulated_mut_effects, all_subs, homologs) = pickle.load(
    open(f"../_ignore/simulated_results_{ps}.pkl", "rb")
)

print(f"\nPlotting")
print(f"--------")

# F() acc
fig, ax = plt.subplots(1, 2, figsize=[8, 5])
sns.scatterplot(data=df, x="observed_phenotype", y="observed_predicted",
                hue="n_aa_substitutions",
                alpha=0.2, palette="deep", ax=ax[0],
                legend=False)

lb = df[["observed_phenotype", "observed_predicted"]].min().min()
ub = df[["observed_phenotype", "observed_predicted"]].max().max()

ax[0].plot([lb, ub], [lb, ub], "k--", lw=1)
r = pearsonr(df.observed_phenotype, df.observed_predicted)[0]
ax[0].annotate(f"$r = {r:.2f}$", (.5, .9), xycoords="axes fraction", fontsize=12)


# Latent acc
#sns.scatterplot(data=df, x="latent_phenotype", y="latent_predicted",
#                hue="n_aa_substitutions",
#                alpha=0.2, palette="deep", ax=ax[1],
#                legend=False)
#
#lb = df[["latent_phenotype", "latent_predicted"]].min().min()
#ub = df[["latent_phenotype", "latent_predicted"]].max().max()
#
#ax[1].plot([lb, ub], [lb, ub], "k--", lw=1)
#r = pearsonr(df.latent_phenotype, df.latent_predicted)[1]
#ax[1].annotate(f"$r = {r:.2f}$", (.5, .9), xycoords="axes fraction", fontsize=12)


# shape 
sns.scatterplot(data=df, x="latent_predicted", y="observed_phenotype",
                hue="n_aa_substitutions",
                alpha=0.2, palette="deep",
                legend=False, ax=ax[1])

#sns.scatterplot(data=df, x="latent_predicted", y="observed_predicted",
#                hue="n_aa_substitutions",
#                alpha=0.2, palette="deep",
#                legend=False, ax=ax[2])


ϕ_grid = onp.linspace(1.1 * df.latent_predicted.min(), 1.1 * df.latent_predicted.max())
shape = (ϕ_grid, g(params["α"], ϕ_grid))
ax[1].plot(*shape)
ax[1].set_ylim(-11, 2)
ax[1].set_xlim(-15, 10)
plt.axhline(0, color="k", ls="--", lw=1)
plt.axvline(0, color="k", ls="--", lw=1)

plt.tight_layout()
fig.savefig(f"../_ignore/eval-scatter-{ps}.png")
print(f"Done")
# load mosaic
# fig, ax = plt.subplots(3, 1, ))
fig = plt.figure(constrained_layout=True, figsize=(20, 20))
axd = fig.subplot_mosaic(
    """
    DDDDDDDDD
    EEEEEEEEE
    FFFFFFFFF
    GGGGGGGGG
    HHHHHHHHH
    """
)
# identify_axes(axd)
"""
"""

layout = {
    "β":("D", "$\hat{β}$"),
    "beta_h1":("E", "$β_{H1}$"),
    "beta_h2":("F", "$β_{H2}$"),
    "S_H2":("G", "$\hat{S}$"),
    "shift":("H", "S")
}

shifted_sites = set(simulated_mut_effects.query("shifted_site == True")["site"])
non_identical_sites = set([
    i+1 for i, (aa1, aa2) in enumerate(zip(*homologs.values()))
    if aa1 != aa2
])

#print(shifted_sites, non_identical_sites)


#for i, param in enumerate(["β"]):
for i, param in enumerate(["β", "S_H2"]):
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

    sns.heatmap(
        mutation_effects, mask=mutation_effects.isnull(),
        linewidths=1,
        cmap="coolwarm_r", center=0,
        # vmin=-1, vmax=1,
        cbar_kws={"label": param},
        ax=axd[layout[param][0]]
    )

    axd[layout[param][0]].set_title(layout[param][1], size=20)


# for i, param in enumerate(["beta_h1"], 2):

for i, param in enumerate(["beta_h1", "beta_h2", "shift"], 2):

    mutation_effects = simulated_mut_effects.pivot(
        index="mut_aa",
        columns="site", values=param
    )

    sns.heatmap(
        mutation_effects, mask=mutation_effects.isnull(),
        linewidths=1,
        cmap="coolwarm_r", center=0,
        # vmin=-1, vmax=1,
        cbar_kws={"label": param},
        ax=axd[layout[param][0]]
    )
    axd[layout[param][0]].set_title(layout[param][1], size=20)


for param, ax in axd.items():
    for site in shifted_sites:
        ax.add_patch(
            plt.Rectangle(
                (site-1, 0), 1, 21, 
                linewidth=3, 
                edgecolor="black" if site in non_identical_sites else "purple", 
                #edgecolor="black",
                fill=False
            )
        )


fig.savefig(f"../_ignore/heatmaps-{ps}.png")
