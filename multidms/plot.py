
import seaborn as sns
import matplotlib.pyplot as plt
import plotnine
from scipy.stats import pearsonr

from functools import reduce
import sys
sys.path.append("..")
from multidms.utils import *

substitution_column = 'aa_substitutions_reference'
experiment_column = 'homolog_exp'
# experiment_column = 'homolog'
scaled_func_score_column = 'log2e'
pal = sns.color_palette('colorblind')


def plot_pred_scatter(
    results, 
    show=True, 
    save=False, 
    annotation_vars=None, 
    printrow=True, 
    annotate_params=False, 
    hue=False
):

    for idx, row in results.iterrows():
    
        if printrow: print(row)
        df = row.variant_prediction_df
        
        def is_wt(string):
            return True if len(string.split()) == 0 else False
        
        df = df.assign(is_wt = df[substitution_column].apply(is_wt))
        fig, ax = plt.subplots(1, 2, figsize=[10, 4], sharey=True)

        sns.scatterplot(
            data=df, x=f"predicted_func_score",
            y=f"corrected_func_score",
            hue=experiment_column if hue else None,
            alpha=0.01, palette="deep", ax=ax[0],
            legend=False
        )

        for group, wt_exp_df in df.query(f"is_wt == True").groupby([experiment_column]):
            wt_pred = wt_exp_df[f"predicted_func_score"].apply(lambda x: round(x, 5)).unique()
            assert len(wt_pred) == 1
            ax[0].axvline(wt_pred[0], label=group)

        lb = df[['func_score', f"predicted_func_score"]].min().min()
        ub = df[['func_score', f"predicted_func_score"]].max().max()

        ax[0].plot([lb, ub], [lb, ub], "k--", lw=1)
        r = pearsonr(df[f'corrected_func_score'], df[f'predicted_func_score'])[0]
        ax[0].annotate(f"$r = {r:.2f}$", (.1, .9), xycoords="axes fraction", fontsize=12)
        ax[0].set_ylabel("functional score")
        ax[0].set_xlabel("predicted functional score")

        if row.model == "non-linear":
            sns.scatterplot(
                data=df, x="predicted_latent_phenotype",
                y=f"corrected_func_score",
                hue=experiment_column if hue else None,
                alpha=0.01, palette="deep",
                legend=True, ax=ax[1]
            )

        if row.model == "non-linear":
            for group, wt_exp_df in df.query(f"is_wt == True").groupby([experiment_column]):
                wt_pred = wt_exp_df[f"predicted_latent_phenotype"].apply(lambda x: round(x, 5)).unique()
                assert len(wt_pred) == 1
                ax[1].axvline(wt_pred[0], label=group)

        if row.model == "non-linear":
            ax[1].legend(loc= "center right", bbox_to_anchor=(1.7, 1.00))
        
        ϕ_grid = onp.linspace(
            1.1 * df.predicted_latent_phenotype.min(),
            1.1 * df.predicted_latent_phenotype.max(),
            num=1000
        )

        if "α" in row.tuned_model_params:
            shape = (ϕ_grid, g(row.tuned_model_params["α"], ϕ_grid))
            ax[1].plot(*shape, color='k', lw=1)
        ax[0].axhline(0, color="k", ls="--", lw=1)
        ax[0].axvline(0, color="k", ls="--", lw=1)
        ax[1].axhline(0, color="k", ls="--", lw=1)
        ax[1].axvline(0, color="k", ls="--", lw=1)
        
        if annotate_params:
            sig_param_anno = ""
            params = row.tuned_model_params
            for param, value in params.items():
                if param[0] not in ["α", "C", "γ"]: continue
                if param == "α":
                    for a_param, a_value in params[param].items():
                        sig_param_anno += f"{a_param}: {round(a_value[0], 2)}\n"
                else:
                    sig_param_anno += f"{param}: {round(value[0], 2)}\n"
            ax[1].annotate(sig_param_anno, (.1, .6), xycoords="axes fraction", fontsize=8)
         
        if annotation_vars:
            annotation_string = "Fit Hyperparams\n------------\n"
            for anno in annotation_vars:
                annotation_string += f"{anno}: {row[anno]}\n"
            ax[1].text(1.1, 0.0, annotation_string, transform=ax[1].transAxes, 
                size=8)

        ax[1].set_xlabel("predicted_latent_phenotype (ϕ)")
        ax[0].set_ylabel("functional score - γ$_{h}$")
        # ax[1].plot(*shape, color='k', lw=1)
        # ax[0].set_ylim(-5, 2.5)       
        ax[1].set_xlim(-11, 6)       
        plt.tight_layout()
        ax[0].grid()
        ax[1].grid()
        if save:
            saveas = "scatter"
            for key, value in row.items():
                if key in [
                    "tuned_model_params", 
                    "all_subs", 
                    "variant_prediction_df", 
                    "site_map"
                ]: continue
                saveas += f"-{value}"
            saveas = "".join(saveas.split()) + ".png"
            fig.savefig(saveas)
        if show: plt.show()


def plot_param_hist(results, show=True, save=False, printrow=False):

    for idx, row in results.iterrows():
        if printrow: print(row)
        mut_effects_df = row.mutation_effects_df
        shift_params = [f"S_{c}" for c in row.conditions[1:]]
        fig, ax = plt.subplots(ncols=(1 + len(shift_params)), figsize=[10,4])
        if len(shift_params) == 0: ax = [ax]
        for (i, param) in enumerate(["β"] + shift_params):

            # Plot data for all mutations
            data = mut_effects_df[mut_effects_df['mut'] != '*']
            bin_width = 0.25
            min_val = math.floor(data[param].min()) - 0.25/2
            max_val = math.ceil(data[param].max())
            sns.histplot(
                x=param, data=data, ax=ax[i],
                stat='density', color=pal.as_hex()[0],
                label='muts to amino acids',
                binwidth=bin_width, binrange=(min_val, max_val),
                alpha=0.5
            )

            # Plot data for mutations leading to stop codons
            data = mut_effects_df[mut_effects_df['mut'] == '*']
            sns.histplot(
                x=param, data=data, ax=ax[i],
                stat='density', color=pal.as_hex()[1],
                label='muts to stop codons',
                binwidth=bin_width, binrange=(min_val, max_val),
                alpha=0.5
            )

            ax[i].set(xlabel=param)
        #axs[1].set_yscale('log')
        plt.tight_layout()
        ax[0].legend()
        if show: plt.show()


def plot_param_heatmap(results, show=True, save=False, printrow=False):

    for idx, row in results.iterrows():
        
        if printrow: print(row)

        shift_params = [f"S_{c}" for c in row.conditions[1:]]
        total_plots = 1 + len(shift_params)
        fig, ax = plt.subplots(total_plots, figsize=[25,(5*total_plots)])
        if len(shift_params) == 0: ax = [ax]
        for (i, param) in enumerate(["β"] + shift_params):

            mut_effects_df = row.mutation_effects_df.copy()
            mut_effects_df.site = mut_effects_df.site.astype(int)

            mutation_effects = mut_effects_df.pivot(
                index="mut",
                columns="site", values=param
            )

            sns.heatmap(
                mutation_effects, 
                mask=mutation_effects.isnull(),
                cmap="coolwarm_r",
                center=0,
                vmin=-1,
                vmax=1,
                cbar_kws={"label": param},
                ax=ax[i]
            )
            ax[i].set_title(f"{param}", size=20)

        plt.tight_layout()
        if show: plt.show()


def plot_param_by_site(results, show=True, save=False, printrow=False):

    for idx, row in results.iterrows():
        if printrow: print(row)
        mut_effects_df = row.mutation_effects_df.copy()
        mut_effects_df.site = mut_effects_df.site.astype(int)

        shift_params = [f"S_{c}" for c in row.conditions[1:]]
        total_plots = 1 + len(shift_params)
        fig, ax = plt.subplots(total_plots, figsize=[12,(3*total_plots)])
        if len(shift_params) == 0: ax = [ax]
        for (i, param) in enumerate(["β"] + shift_params):

            mutation_effects = mut_effects_df.pivot(
                index="mut",
                columns="site", values=param
            ).apply(lambda x: sum([abs(t) for t in x if t == t]), axis=0).reset_index()

            ax[i].axhline(0, color="k", ls="--", lw=1)

            # TODO kwargs?
            #ax[i].set_ylim([-1, 50])
            #ax[i].set_xlim([830, 860])

            sns.lineplot(
                data=mutation_effects,
                x="site", y=0, 
                ax=ax[i]
            )

            # TODO
            #non_identical_sites = [
            #    int(i) for i, s in row.site_map.iterrows()
            #    if s[row.experiment_ref] != s[row.experiment_2]
            #]
            #mutation_effects["non_identical"] = [
            #    True if s in non_identical_sites else False for s in mutation_effects.site
            #]

            sns.scatterplot(
                data=mutation_effects,
                x="site", y=0,
                #hue = "non_identical",
                ax=ax[i]
            )
            # TODO more informative ylabel abs value
            ax[i].set_ylabel(f"$\sum${param}", size=10)

        plt.tight_layout()
        if show: plt.show()


def plot_fit_param_comp_scatter(
    results,
    fits_features,
    show=True
):

    mut_effect_dfs = []
    fit_features = []
    for fit, feature in fits_features.items():
        fit_mut_effects = results.loc[fit, "mutation_effects_df"].copy()
        fit_mut_effects.site = fit_mut_effects.site.astype(int)
        mut_effect_dfs.append(fit_mut_effects)
        fit_features.append(feature)

    comb_mut_effects = reduce(
        lambda l, r: pd.merge(
            l, r, how="inner", on="substitution"
        ), mut_effect_dfs
    )
    comb_mut_effects["is_stop"] = [
        True if "*" in s else False for s in comb_mut_effects.substitution
    ]

    if fit_features[0] == fit_features[1]:
        x = f"{fit_features[0]}_x"
        y = f"{fit_features[1]}_y"
    else:
        x = fit_features[0]
        y = fit_features[1] 

    fig, ax = plt.subplots(figsize=[5,4])
    r = pearsonr(comb_mut_effects[x], comb_mut_effects[y])[0]
    sns.scatterplot(
        data=comb_mut_effects, 
        x=x, y=y,
        hue="is_stop", 
        alpha=0.6,  
        palette="deep", 
        ax=ax
    )

    min1, max1 = comb_mut_effects[x].min(), comb_mut_effects[x].max()
    ax.plot([min1, max1], [min1, max1], ls="--", c="k")
    ax.annotate(f"$r = {r:.2f}$", (.7, .1), xycoords="axes fraction", fontsize=12)
    ax.set_title(f"feature comparison")
    plt.tight_layout()
    if show: plt.show()


def plot_fit_param_site_comp_scatter(results, idx_1, idx_2, show=True, save=False, printrow=False):

    def split_sub(sub_string):
        """String match the wt, site, and sub aa
        in a given string denoting a single substitution"""

        pattern = r'(?P<aawt>[A-Z\*])(?P<site>[\d\w]+)(?P<aamut>[A-Z\*])'
        match = re.search(pattern, sub_string)
        assert match != None, sub_string
        return match.group('aawt'), match.group('site'), match.group('aamut')

    dfs = []
    for fit in [idx_1, idx_2]:

        fit_exp2 = results.loc[fit, "experiment_2"]
        param = f"S_{fit_exp2}"
        fit_subs = results.loc[fit, "all_subs"]
        fit_shifts = results.loc[fit, "tuned_model_params"][param]
        rows = []
        for mutation, p in zip(fit_subs, fit_shifts):
            wt, site, mut = split_sub(mutation)
            rows.append([int(site), wt, mut, float(p)])

        mutation_effects = pd.DataFrame(
            rows,
            columns=("site", "wildtype", "mutant", param)
        ).pivot(
            index="mutant",
            columns="site", values=param
        ).apply(
            lambda x: sum([abs(t) for t in x if t == t]), axis=0
        ).reset_index().rename({0:param}, axis=1)

        dfs.append(pd.DataFrame(
            mutation_effects
        ))
    df = reduce(
        lambda l, r: pd.merge(
            l, r, how="inner", on="site"
        ), dfs
    )
    r = pearsonr(df.iloc[:, 1], df.iloc[:, 2])[0]
    fig, ax = plt.subplots(figsize=[6, 6])
    sns.scatterplot(
        data=df, 
        x=df.columns[1], 
        y=df.columns[2], 
        alpha=0.6,  
        palette="deep", 
        ax=ax
    )
    fit1_ref, fit1_e2 = results.loc[idx_1, ["experiment_ref", "experiment_2"]].values
    fit2_ref, fit2_e2 = results.loc[idx_2, ["experiment_ref", "experiment_2"]].values
    min1, max1 = df.iloc[:,1].min(), df.iloc[:,1].max()
    ax.plot([min1, max1], [min1, max1], ls="--", c="k")
    ax.annotate(f"$r = {r:.2f}$", (.7, .1), xycoords="axes fraction", fontsize=12)
    ax.set_title(f"Shift parameter comparison")
    ax.set_xlabel(f"S({fit1_ref} -> {fit1_e2})")
    ax.set_ylabel(f"S({fit2_ref} -> {fit2_e2})")

    plt.tight_layout()
    if save:
        saveas = "param-heatmap"
        for key, value in row.items():
            if key in ["tuned_model_params", "all_subs", "variant_prediction_df", "site_map"]: continue
            saveas += f"-{value}"
        saveas = "".join(saveas.split()) + ".png"
        fig.savefig(saveas)

    if show: plt.show()
