
import seaborn as sns
import matplotlib.pyplot as plt
import plotnine
from scipy.stats import pearsonr

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
            data=df, x=f"predicted_{row.func_score_target}",
            y=row.func_score_target,
            hue=experiment_column if hue else None,
            alpha=0.01, palette="deep", ax=ax[0],
            legend=False
        )
        for group, wt_exp_df in df.query(f"is_wt == True").groupby([experiment_column]):
            wt_pred = wt_exp_df[f"predicted_{row.func_score_target}"].apply(lambda x: round(x, 5)).unique()
            assert len(wt_pred) == 1
            ls = "-" if group == row.experiment_ref else "--"
            co = "blue" if group == row.experiment_ref else "orange"
            ax[0].axvline(wt_pred[0], color=co, ls=ls, label=group)

        lb = df[[row.func_score_target, f"predicted_{row.func_score_target}"]].min().min()
        ub = df[[row.func_score_target, f"predicted_{row.func_score_target}"]].max().max()

        ax[0].plot([lb, ub], [lb, ub], "k--", lw=1)
        r = pearsonr(df[row.func_score_target], df[f'predicted_{row.func_score_target}'])[0]
        ax[0].annotate(f"$r = {r:.2f}$", (.1, .9), xycoords="axes fraction", fontsize=12)
        ax[0].set_ylabel("functional score")
        ax[0].set_xlabel("predicted functional score")


        idx = df.query(f"{experiment_column} == '{row.experiment_2}'").index 
        df_ep = df.copy()
        if row.model == "non-linear":
            df_ep.loc[idx, row.func_score_target] -= row.tuned_model_params[f"γ_{row.experiment_2}"][0]
            # linear adjustment?
            #df_ep.loc[idx, "predicted_latent_phenotype"] += row.tuned_model_params[f"γ_{row.experiment_2}"][0]
            sns.scatterplot(
                data=df_ep, x="predicted_latent_phenotype",
                y=row.func_score_target,
                # hue=experiment_column,
                hue=experiment_column if hue else None,
                alpha=0.01, palette="deep",
                legend=True, ax=ax[1]
            )
        if row.model == "non-linear":
            for group, wt_exp_df in df.query(f"is_wt == True").groupby([experiment_column]):
                wt_pred = wt_exp_df[f"predicted_latent_phenotype"].apply(lambda x: round(x, 5)).unique()
                assert len(wt_pred) == 1
                ls = "-" if group == row.experiment_ref else "--"
                co = "blue" if group == row.experiment_ref else "orange"
                ax[1].axvline(wt_pred[0], color=co, ls=ls, label=f"{group} wt")

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
        # ax[1].plot(*shape, color='k', lw=1)
        # ax[1].set_ylim(-4, 2.5)       
        #ax[0].set_xlim(-5, 2.5)       
        #ax[0].set_ylim(-5, 3)       
        #ax[1].set_xlim(-11, 6)       
        #ax[1].set_ylim(-5, 3)       
        plt.tight_layout()
        if save:
            saveas = "scatter"
            for key, value in row.items():
                if key in ["tuned_model_params", "all_subs", "variant_prediction_df", "site_map"]: continue
                saveas += f"-{value}"
            saveas = "".join(saveas.split()) + ".png"
            fig.savefig(saveas)
        if show: plt.show()


def plot_param_hist(results, show=True, save=False, printrow=False):
    for idx, row in results.iterrows():
        # Make a dataframe with inferred mutational effects
        
        if printrow: print(row)
        
        def split_sub(sub_string):
            """String match the wt, site, and sub aa
            in a given string denoting a single substitution"""

            pattern = r'(?P<aawt>[A-Z\*])(?P<site>[\d\w]+)(?P<aamut>[A-Z\*])'
            match = re.search(pattern, sub_string)
            assert match != None, sub_string
            return match.group('aawt'), match.group('site'), match.group('aamut')

        mut_effects_dict = {
            key : []
            for key in ['wt', 'site', 'mut', 'param', 'param_val']
        }
        for (i, param) in enumerate(["β", f"S_{row.experiment_2}"]):
            if param in row.tuned_model_params:
                for (mutation, param_val) in zip(row.all_subs, row.tuned_model_params[param]):
                    (wt, site, mut) = split_sub(mutation)
                    mut_effects_dict['wt'].append(wt)
                    mut_effects_dict['site'].append(int(site))
                    mut_effects_dict['mut'].append(mut)
                    mut_effects_dict['param'].append(param)
                    mut_effects_dict['param_val'].append(float(param_val))

        mut_effects_df = pd.DataFrame(mut_effects_dict)

        (fig, axs) = plt.subplots(ncols=2, figsize=[10,4])
        for (i, param) in enumerate(["β", f"S_{row.experiment_2}"]):

            # Plot data for all mutations
            data = mut_effects_df[
                (mut_effects_df['mut'] != '*') &
                (mut_effects_df['param'] == param)
            ]
            bin_width = 0.25
            min_val = math.floor(data['param_val'].min()) - 0.25/2
            max_val = math.ceil(data['param_val'].max())
            sns.histplot(
                x='param_val', data=data, ax=axs[i],
                stat='density', color=pal.as_hex()[0],
                label='muts to amino acids',
                binwidth=bin_width, binrange=(min_val, max_val),
                alpha=0.5
            )

            # Plot data for mutations leading to stop codons
            data = mut_effects_df[
                (mut_effects_df['mut'] == '*') &
                (mut_effects_df['param'] == param)
            ]
            sns.histplot(
                x='param_val', data=data, ax=axs[i],
                stat='density', color=pal.as_hex()[1],
                label='muts to stop codons',
                binwidth=bin_width, binrange=(min_val, max_val),
                alpha=0.5
                #log_scale=True
            )

            axs[i].set(xlabel=param)
        #axs[1].set_yscale('log')
        plt.tight_layout()
        axs[0].legend()
        if save:
            saveas = "param-hist"
            for key, value in row.items():
                if key in ["tuned_model_params", "all_subs", "variant_prediction_df", "site_map"]: continue
                saveas += f"-{value}"
            saveas = "".join(saveas.split()) + ".png"
            fig.savefig(saveas)
        if show: plt.show()


def plot_param_heatmap(results, show=True, save=False, printrow=False):

    for idx, row in results.iterrows():
        
        if printrow: print(row)
        fig, ax = plt.subplots(2, figsize=(25, 10))

        # TODO add outlining plots
        # non_identical_sites = [
        #     i for i, row in site_map.iterrows()
        #     if row["Delta"] != row["Omicron_BA.1"]
        # ]

        def split_sub(sub_string):
            """String match the wt, site, and sub aa
            in a given string denoting a single substitution"""

            pattern = r'(?P<aawt>[A-Z\*])(?P<site>[\d\w]+)(?P<aamut>[A-Z\*])'
            match = re.search(pattern, sub_string)
            assert match != None, sub_string
            return match.group('aawt'), match.group('site'), match.group('aamut')

        for i, param in enumerate(["β", f"S_{row.experiment_2}"]):
            rows = []
            for mutation, p in zip(row.all_subs, row.tuned_model_params[param]):
                wt, site, mut = split_sub(mutation)
                rows.append([int(site), wt, mut, float(p)])

            mutation_effects = pd.DataFrame(
                rows,
                columns=("site", "wildtype", "mutant", param)
            ).pivot(
                index="mutant",
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
        if save:
            saveas = "param-heatmap"
            for key, value in row.items():
                if key in ["tuned_model_params", "all_subs", "variant_prediction_df", "site_map"]: continue
                saveas += f"-{value}"
            saveas = "".join(saveas.split()) + ".png"
            fig.savefig(saveas)
        if show: plt.show()


def plot_shift_by_site(results, show=True, save=False, printrow=False):

    for idx, row in results.iterrows():
        
        if printrow: print(row)
        fig, ax = plt.subplots(2, figsize=(15, 5))

        non_identical_sites = [
            int(i) for i, s in row.site_map.iterrows()
            if s[row.experiment_ref] != s[row.experiment_2]
        ]
        # print(non_identical_sites)

        def split_sub(sub_string):
            """String match the wt, site, and sub aa
            in a given string denoting a single substitution"""

            pattern = r'(?P<aawt>[A-Z\*])(?P<site>[\d\w]+)(?P<aamut>[A-Z\*])'
            match = re.search(pattern, sub_string)
            assert match != None, sub_string
            return match.group('aawt'), match.group('site'), match.group('aamut')

        for i, param in enumerate(["β", f"S_{row.experiment_2}"]):
            rows = []
            for mutation, p in zip(row.all_subs, row.tuned_model_params[param]):
                wt, site, mut = split_sub(mutation)
                rows.append([int(site), wt, mut, float(p)])

            mutation_effects = pd.DataFrame(
                rows,
                columns=("site", "wildtype", "mutant", param)
            ).pivot(
                index="mutant",
                columns="site", values=param
            ).apply(lambda x: sum([abs(t) for t in x if t == t]), axis=0).reset_index()

            ax[i].axhline(0, color="k", ls="--", lw=1)

            sns.lineplot(
                data=mutation_effects,
                x="site", y=0, 
                ax=ax[i]
            )

            mutation_effects["non_identical"] = [True if s in non_identical_sites else False for s in mutation_effects.site]
            sns.scatterplot(
                data=mutation_effects,
                # size=1,
                x="site", y=0,
                hue = "non_identical",
                ax=ax[i]
            )
            ax[i].set_ylabel(f"$\sum${param[:9]}", size=20)

        plt.tight_layout()
        if save:
            saveas = "param-heatmap"
            for key, value in row.items():
                if key in ["tuned_model_params", "all_subs", "variant_prediction_df", "site_map"]: continue
                saveas += f"-{value}"
            saveas = "".join(saveas.split()) + ".png"
            fig.savefig(saveas)
        if show: plt.show()


def plot_fit_param_comp_scatter(results, idx_1, idx_2, show=True, save=False, printrow=False):

    to_compare = ["tuned_model_params", "all_subs"] 
    fit1 = results.loc[idx_1, to_compare]
    fit2 = results.loc[idx_2, to_compare]

    # make sure we're comparing the same substitution parameters
    assert onp.all(fit1.all_subs == fit2.all_subs)
    
    fig, ax = plt.subplots(2, figsize=[6, 10])
    i=0
    for param in fit1.tuned_model_params:
        if param == "β" or param == f"S_{results.loc[idx_1, 'experiment_2']}":
            df = pd.DataFrame(
                {
                    "fit1": fit1.tuned_model_params[param],
                    "fit2": fit2.tuned_model_params[param],
                    "is_stop":[True if "*" in s else False for s in fit1.all_subs]
                }
            )
            sns.scatterplot(data=df, x="fit1", y="fit2", hue="is_stop", alpha=0.6,  palette="deep", ax=ax[i])
            ax[i].plot([df.fit1.min(), df.fit1.max()], [df.fit1.min(), df.fit1.max()], ls="--", c="k")
            ax[i].set_title(f"{param} parameter comparison")
            i += 1

    plt.tight_layout()
    if save:
        saveas = "param-heatmap"
        for key, value in row.items():
            if key in ["tuned_model_params", "all_subs", "variant_prediction_df", "site_map"]: continue
            saveas += f"-{value}"
        saveas = "".join(saveas.split()) + ".png"
        fig.savefig(saveas)
    if show: plt.show()
