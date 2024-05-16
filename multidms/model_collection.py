"""
Contains the ModelCollection class, which takes a collection of models
and merges the results for comparison and visualization.
"""

import itertools as it
import time
import warnings
from functools import lru_cache
from multiprocessing import get_context
import multiprocessing
import pprint
import logging

import multidms

import pandas as pd
import jax

import numpy as onp
import altair as alt

jax.config.update("jax_enable_x64", True)


logging.getLogger("jax._src.xla_bridge").addFilter(
    logging.Filter(
        "An NVIDIA GPU may be present on this machine, "
        "but a CUDA-enabled jaxlib is not installed. Falling back to cpu."
    )
)


PARAMETER_NAMES_FOR_PLOTTING = {
    "scale_coeff_lasso_shift": "Lasso Penalty",
}


class ModelCollectionFitError(Exception):
    """Error fitting models."""

    pass


# TODO move to utils
# https://github.com/microsoft/pylance-release/issues/5630
def my_concat(dfs_list, axis=0):
    """simple pd.concat wrapper for bug with vscode pylance"""  # noqa
    return pd.concat(dfs_list, axis=axis)


# TODO move to utils
def _explode_params_dict(params_dict):
    """
    Given a dictionary of model parameters,
    of which any of the values can be a list of values,
    compute all combinations of model parameter sets
    and returns a list of dictionaries representing each
    of the parameter sets.
    """
    varNames = sorted(params_dict)
    return [
        dict(zip(varNames, prod))
        for prod in it.product(*(params_dict[varName] for varName in varNames))
    ]


# TODO could kwargs be used both for init and fit of model?
def fit_one_model(
    dataset,
    epistatic_model="Sigmoid",
    output_activation="Identity",
    # gamma_corrected=False,  # TODO GAMMA experimental, mark as such
    init_theta_scale=6.5,
    init_theta_bias=-3.5,
    n_hidden_units=5,
    lower_bound=None,
    PRNGKey=0,
    verbose=False,
    **kwargs,
):
    """
    Fit a multidms model to a dataset. This is a wrapper around the multidms
    fit method that allows for easy specification of the fit parameters.
    This method is helpful for comparing and organizing multiple fits.

    Parameters
    ----------
    dataset : :class:`multidms.Data`
        The dataset to fit to. For bookkeeping and downstream analysis,
        the name of the dataset (Data.name) is saved in the fit attributes
        that are returned.
    epistatic_model : str, optional
        The epistatic model to use. The default is "Identity".
    output_activation : str, optional
        The output activation function to use. The default is "Identity".
    init_theta_scale : float, optional
        The scale to use for initializing the model parameters. The default is 6.5.
    init_theta_bias : float, optional
        The bias to use for initializing the model parameters. The default is -3.5.
    n_hidden_units : int, optional
        The number of hidden units to use in the neural network model. The default is 5.
    lower_bound : float, optional
        The lower bound for use with the softplus activation function.
        The default is None, but must be specified if using the softplus activation.
    PRNGKey : int, optional
        The PRNGKey to use to initialize model parameters. The default is 0.
    verbose : bool, optional
        Whether to print out information about the fit to stdout. The default is False.
    **kwargs : dict
        Additional keyword arguments to pass to the multidms.Model.fit method.

    Returns
    -------
    fit_series : :class:`pandas.Series`
        A series containing reference to the fit `multidms.Model` object
        and the associated parameters used for the fit.
        These consist mostly of the keyword arguments passed to this function,
        less "verbose", and with the addition of:
        1. "model" - the fit `multidms.Model` object reference,
        2. "dataset_name" which will simply be the name associated with the `Data`
        object
        used for training (note that the `multidms.Data` object itself is always
        accessible via the `Model.data` attribute).
        3. "step_loss" which is a numpy array of the loss at the end of each training
        epoch.
    """
    # save the passed parameters for bookkeeping
    fit_attributes = locals().copy()
    del fit_attributes["kwargs"]
    for key, value in kwargs.items():
        fit_attributes[key] = value

    biophysical_model = {
        "Identity": multidms.biophysical.identity_activation,
        "Sigmoid": multidms.biophysical.sigmoidal_global_epistasis,
        "NN": multidms.biophysical.nn_global_epistasis,
        "Softplus": multidms.biophysical.softplus_activation,
    }

    # should these all be kwargs?
    imodel = multidms.Model(
        dataset,
        epistatic_model=biophysical_model[epistatic_model],
        output_activation=biophysical_model[output_activation],
        init_theta_scale=init_theta_scale,
        init_theta_bias=init_theta_bias,
        # gamma_corrected=gamma_corrected, TODO GAMMA
        n_hidden_units=n_hidden_units,
        lower_bound=lower_bound,
        PRNGKey=PRNGKey,
    )

    del fit_attributes["dataset"]
    del fit_attributes["verbose"]

    fit_attributes["dataset_name"] = dataset.name
    fit_attributes["model"] = imodel

    if verbose:
        print("running:")
        pprint.pprint(fit_attributes)

    # for training_step in range(num_training_steps):
    start = time.time()
    imodel.fit(
        warn_unconverged=verbose,
        **kwargs,
    )
    end = time.time()

    fit_attributes["fit_time"] = round(end - start)

    return pd.Series(fit_attributes)


def _fit_fun(params):
    """Workaround for multiprocessing to fit models with sets of kwargs"""
    _, kwargs = params
    try:
        return fit_one_model(**kwargs)
    except Exception:
        return None


def stack_fit_models(fit_models_list):
    """
    given a list of pd.Series objects returned by fit_one_model,
    stack them into a single pd.DataFrame
    """
    return pd.concat([f.to_frame().T for f in fit_models_list], ignore_index=True)


# TODO make it easier to debug failed fits
def fit_models(params, n_threads=-1, failures="error"):
    """Fit collection of :class:`multidms.model.Model` models.

    Enables fitting of multiple models simultaneously using multiple threads.
    Most commonly, this function is used to fit a set of models across combinations
    of replicate training datasets, and lasso coefficients for model selection and
    evaluation. The returned dataframe is meant to be passed into the
    :class:`multidms.model_collection.ModelCollection` class for comparison
    and visualization.

    Parameters
    ----------
    params : dict
        Dictionary which defines the parameter space of all models you
        wish to run. Each value in the dictionary must be a list of
        values, even in the case of singletons.
        This function will compute all combinations of the parameter
        space and pass each combination to :func:`multidms.utils.fit_one_model`
        to be run in parallel, thus only key-value pairs which
        match the kwargs are allowed.
        See the docstring of :func:`multidms.model_collection.fit_one_model` for
        a description of the allowed parameters.
    n_threads : int
        Number of threads (CPUs, cores) to use for fitting. Set to -1 to use
        all CPUs available.
    failures : {"error", "tolerate"}
        What if fitting fails for a model? If "error" then raise an error,
        if "ignore" then just return `None` for models that failed optimization.

    Returns
    -------
    (n_fit, n_failed, fit_models)
        Number of models that fit successfully, number of models that failed,
        and a dataframe which contains a row for each of the `multidms.Model`
        object references along with the parameters each was fit with for convenience.
        The dataframe is ultimately meant to be passed into the ModelCollection class.
        for comparison and visualization.
    """
    if n_threads == -1:
        n_threads = multiprocessing.cpu_count()

    exploded_params = _explode_params_dict(params)
    # if __name__ == "__main__":
    # see https://pythonspeed.com/articles/python-multiprocessing/ for why we spawn
    with get_context("spawn").Pool(n_threads) as p:
        fit_models = p.map(_fit_fun, [(None, params) for params in exploded_params])
        # fit_models = p.map(
        #     _fit_fun, [(params.pop("dataset"), params) for params in exploded_params]
        # )
        # p.close()
    assert len(fit_models) == len(exploded_params)

    # Check to see if any models failed optimization
    n_failed = sum(model is None for model in fit_models)
    if failures == "error":
        if n_failed:
            raise ModelCollectionFitError(
                f"Failed fitting {n_failed} of {len(exploded_params)} parameter sets"
            )
    elif failures != "tolerate":
        raise ValueError(f"invalid {failures=}")
    n_fit = len(fit_models) - n_failed
    if n_fit == 0:
        raise ModelCollectionFitError(
            f"Failed fitting all {len(exploded_params)} parameter sets"
        )

    return (n_fit, n_failed, stack_fit_models(fit_models))


class ModelCollection:
    """
    A class for the comparison and visualization of multiple
    `multidms.Model` fits. The respective collection of
    training datasets for each fit must
    share the same reference sequence and conditions. Additionally,
    the inferred site maps must agree upon condition wildtypes
    for all shared sites.

    The utility function `multidms.model_collection.fit_models` is used to fit
    the collection of models, and the resulting dataframe is passed to the
    constructor of this class.

    Parameters
    ----------
    fit_models : :class:`pandas.DataFrame`
        A dataframe containing the fit attributes and pickled model objects
        as returned by `multidms.model_collection.fit_models`.
    """

    def __init__(self, fit_models):
        """See class docstring."""
        # Check that all datasets share reference, and conditions, and site maps
        first_dataset = fit_models.iloc[0].model.data
        validated_datasets = [first_dataset.name]
        site_map_union = first_dataset.site_map.copy()
        shared_mutations = set(first_dataset.mutations)
        all_mutations = set(first_dataset.mutations)
        for fit in fit_models.model:
            if fit.data.name in validated_datasets:
                continue
            if fit.data.reference != first_dataset.reference:
                raise ValueError(
                    "All model training datasets must share the same reference sequence"
                )
            if not len(set(fit.data.conditions) - set(first_dataset.conditions)) == 0:
                raise ValueError(
                    "All model training datasets must share the same conditions"
                )
            shared_sites = list(
                set.intersection(
                    set(site_map_union.index), set(fit.data.site_map.index)
                )
            )

            if not site_map_union.loc[shared_sites].equals(
                fit.data.site_map.loc[shared_sites]
            ):
                raise ValueError(
                    "All model training datasets must share the same site map"
                )
            new_sites = list(set(fit.data.site_map.index) - set(site_map_union.index))
            if len(new_sites) > 0:
                site_map_union = pd.concat(
                    [site_map_union, fit.data.site_map.loc[new_sites]]
                ).sort_index()
            validated_datasets.append(fit.data.name)

            shared_mutations = set.intersection(
                shared_mutations, set(fit.data.mutations)
            )
            all_mutations = set.union(all_mutations, set(fit.data.mutations))

        # Add convergence flag.
        fit_models["converged"] = fit_models.model.apply(lambda x: x.converged).astype(
            bool
        )
        for idx, fit in fit_models.iterrows():
            conditional_loss = fit.model.conditional_loss
            for condition, loss in conditional_loss.items():
                fit_models.loc[idx, f"{condition}_loss_training"] = loss

        self._site_map_union = site_map_union
        self._conditions = first_dataset.conditions
        self._reference = first_dataset.reference
        self.fit_models = fit_models
        self.condition_colors = first_dataset.condition_colors
        self._shared_mutations = tuple(shared_mutations)
        self._all_mutations = tuple(all_mutations)

    @property
    def site_map_union(self) -> pd.DataFrame:
        """The union of all site maps of all datasets used for fitting."""
        return self._site_map_union

    @property
    def conditions(self) -> list:
        """The conditions (shared by each fitting dataset) used for fitting."""
        return self._conditions

    @property
    def reference(self) -> str:
        """The reference conditions (shared by each fitting dataset) used for fitting."""
        return self._reference

    @property
    def shared_mutations(self) -> tuple:
        """The mutations shared by each fitting dataset."""
        return self._shared_mutations

    @property
    def all_mutations(self) -> tuple:
        """The mutations shared by each fitting dataset."""
        return self._all_mutations

    @lru_cache(maxsize=10)
    def split_apply_combine_muts(
        self,
        groupby=("dataset_name", "scale_coeff_lasso_shift"),
        aggregate_func="mean",
        inner_merge_dataset_muts=True,
        query=None,
        **kwargs,
    ):
        """
        wrapper to split-apply-combine the set of mutational dataframes
        harbored by each of the fits in the collection.

        Here, we group the collection of fits using attributes
        (columns in :py:attr:`ModelCollection.fit_models`) specified using the
        ``groupby`` parameter.
        Each of the individual fits within a groups may then be filtered
        via ``**kwargs``, and aggregated via ``aggregate_func``, before
        the function stacks all the groups back together in a
        tall style dataframe. The resulting dataframe will have a multiindex
        with the mutation and the groupby attributes.

        Parameters
        ----------
        groupby : str or tuple of str or None, optional
            The attributes to group the fits by. If None, then group by all
            attributes except for the model, data, and step_loss attributes.
            The default is ("dataset_name", "scale_coeff_lasso_shift").
        aggregate_func : str or callable, optional
            The function to aggregate the mutational dataframes within each group.
            The default is "mean".
        inner_merge_dataset_muts : bool, optional
            Whether to toss mutations which are _not_ shared across all datasets
            before aggregation of group mutation parameter values.
            The default is True.
        query : str, optional
            The pandas query to apply to the `ModelCollection.fit_models`
            dataframe before splitting. The default is None.
        **kwargs : dict
            Keyword arguments to pass to the :func:`multidms.Model.get_mutations_df`
            method ("phenotype_as_effect", and "times_seen_threshold") see the
            method docstring for details.

        Returns
        -------
        :class:`pandas.DataFrame`
            A dataframe containing the aggregated mutational parameter values
        """
        print("cache miss - this could take a moment")
        queried_fits = (
            self.fit_models.query(query) if query is not None else self.fit_models
        )
        if len(queried_fits) == 0:
            raise ValueError("invalid query, no fits returned")

        if groupby is None:
            ret = (
                pd.concat(
                    [
                        fit["model"].get_mutations_df(return_split=False, **kwargs)
                        for _, fit in queried_fits.iterrows()
                    ],
                    join="inner",  # the columns will always match based on class req.
                )
                .query(
                    f"mutation.isin({list(self.shared_mutations)})"
                    if inner_merge_dataset_muts
                    else "mutation.notna()"
                )
                .groupby("mutation")
                .aggregate(aggregate_func)
            )
            return ret

        elif isinstance(groupby, str):
            groupby = tuple([groupby])

        elif isinstance(groupby, tuple):
            if not all(feature in queried_fits.columns for feature in groupby):
                raise ValueError(
                    f"invalid groupby, values must be in {self.fit_models.columns}"
                )
        else:
            raise ValueError(
                "invalid groupby, must be tuple with values "
                f"in {queried_fits.columns}"
            )

        ret = pd.concat(
            [
                pd.concat(
                    [
                        fit["model"].get_mutations_df(return_split=False, **kwargs)
                        for _, fit in fit_group.iterrows()
                    ],
                    join="inner",  # the columns will always match based on class req.
                )
                .query(
                    f"mutation.isin({list(self.shared_mutations)})"
                    if inner_merge_dataset_muts
                    else "mutation.notna()"
                )
                .groupby("mutation")
                .aggregate(aggregate_func)
                .assign(**dict(zip(list(groupby), group)))
                .reset_index()
                .set_index(list(groupby))
                for group, fit_group in queried_fits.groupby(
                    list(groupby), observed=True
                )
            ],
            join="inner",
        )

        return ret

    def add_validation_loss(self, test_data, overwrite=False):
        """
        Add validation loss to the fit collection dataframe.

        Parameters
        ----------
        test_data : pd.DataFrame or dict(str, pd.DataFrame)
            The testing dataframe to compute validation loss with respect to,
            must have columns "aa_substitutitions", "condition", and "func_score".
            If a dictionary is passed, there should be a key for
            each unique dataset_name factor in the self.fit_models dataframe
            - with the value being the respective testing dataframe.
        overwrite : bool, optional
            Whether to overwrite the validation_loss column if it already exists.
            The default is False.

        Returns
        -------
        pd.DataFrame
            The self.fit_models dataframe with the validation loss added.
        """
        if isinstance(test_data, pd.DataFrame):
            temp_test_data = test_data.copy()
            test_data = {}
            for name in self.fit_models["dataset_name"].unique():
                test_data[name] = temp_test_data

        # check there's a testing dataframe for each unique dataset_name
        assert set(test_data.keys()) == set(self.fit_models["dataset_name"].unique())

        validation_cols_exist = onp.any(
            [
                f"{condition}_loss_validation" in self.fit_models.columns
                for condition in self.conditions
            ]
        )
        if validation_cols_exist and not overwrite:
            raise ValueError(
                "validation_loss already exists in self.fit_models, set overwrite=True "
                "to overwrite"
            )

        self.fit_models = self.fit_models.assign(
            **{
                f"{condition}_loss_validation": onp.nan for condition in self.conditions
            },
            total_loss_validation=onp.nan,
        )

        for idx, fit in self.fit_models.iterrows():
            condional_df_loss = fit.model.get_df_loss(
                test_data[fit["dataset_name"]], conditional=True
            )
            for condition, loss in condional_df_loss.items():
                self.fit_models.loc[idx, f"{condition}_loss_validation"] = loss

        return None

    def get_conditional_loss_df(self, query=None):
        """
        return a long form dataframe with columns
        "dataset_name", "scale_coeff_lasso_shift",
        "split" ("training" or "validation"),
        "loss" (actual value), and "condition".

        Parameters
        ----------
        query : str, optional
            The query to apply to the fit_models dataframe
            before formatting the loss dataframe. The default is None.
        """
        if query is not None:
            queried_fits = self.fit_models.query(query)
        else:
            queried_fits = self.fit_models
        if len(queried_fits) == 0:
            raise ValueError("invalid query, no fits returned")

        id_vars = ["dataset_name", "scale_coeff_lasso_shift"]
        value_vars = [
            c for c in queried_fits.columns if "loss" in c and c != "step_loss"
        ]
        loss_df = queried_fits.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="condition",
            value_name="loss",
        ).assign(
            split=lambda x: x.condition.str.split("_").str.get(-1),
            condition=lambda x: x.condition.str.split("_").str[:-2].str.join("_"),
        )
        return loss_df

    def convergence_trajectory_df(
        self,
        query=None,
        id_vars=("dataset_name", "scale_coeff_lasso_shift"),
    ):
        """TODO"""
        queried_fits = (
            self.fit_models.query(query) if query is not None else self.fit_models
        )
        if len(queried_fits) == 0:
            raise ValueError("invalid query, no fits returned")

        if not all([var in queried_fits.columns for var in id_vars]):
            raise ValueError(f"invalid {id_vars=}")

        convergence_trajectory_data = my_concat(
            [
                (
                    fit.model.convergence_trajectory_df.assign(
                        **{key: fit[key] for key in id_vars}
                    )
                )
                for _, fit in queried_fits.iterrows()
            ]
        )

        # TODO make altair chart
        return convergence_trajectory_data

    def mut_param_heatmap(
        self,
        query=None,
        mut_param="shift",
        aggregate_func="mean",
        inner_merge_dataset_muts=True,
        times_seen_threshold=0,
        phenotype_as_effect=True,
        **kwargs,
    ):
        """
        Create lineplot and heatmap altair chart
        across replicate datasets.
        This function optionally applies a given `pandas.query`
        on the fit_models dataframe that should result in a subset of
        fit's which make sense to aggregate mutational data across, e.g.
        replicate datasets.
        It then computes the mean or median mutational parameter value
        ("beta", "shift", or "predicted_func_score")
        between the remaining fits. and creates an interactive altair chart.


        Note that this will throw an error if the queried fits have more
        than one unique hyper-parameter besides "dataset_name".


        Parameters
        ----------
        query : str
            The query to apply to the fit_models dataframe. This should be
            used to subset the fits to only those which make sense to aggregate
            mutational data across, e.g. replicate datasets.
            For example, if you have a collection of
            fits with different epistatic models, you may want to query
            for only those fits with the same epistatic model. e.g.
            `query="epistatic_model == 'Sigmoid'"`. For more on the query
            syntax, see the
            `pandas.query <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html>`_
            documentation.
        mut_param : str, optional
            The mutational parameter to plot. The default is "shift".
            Must be one of "shift", "predicted_func_score", or "beta".
        aggregate_func : str, optional
            The function to aggregate the mutational parameter values
            between dataset fits. The default is "mean".
        inner_merge_dataset_muts : bool, optional
            Whether to toss mutations which are _not_ shared across all datasets
            before aggregation of group mutation parameter values.
            The default is True.
        times_seen_threshold : int, optional
            The minimum number of times a mutation must be seen across
            all conditions within a single fit to be included in the
            aggregation. The default is 0.
        phenotype_as_effect : bool, optional
            Passed to `Model.get_mutations_df()`,
            Only applies if `mut_param="predicted_func_score"`.
        **kwargs : dict
            Keyword arguments to pass to
            :func:`multidms.plot._lineplot_and_heatmap`.

        Returns
        -------
        altair.Chart
            A chart object which can be displayed in a jupyter notebook
            or saved to a file.
        """
        queried_fits = (
            self.fit_models.query(query) if query is not None else self.fit_models
        )
        if len(queried_fits) == 0:
            raise ValueError("invalid query, no fits returned")
        shouldbe_uniform = list(
            set(queried_fits.columns)
            - set(
                ["model", "dataset_name"]
                + [col for col in queried_fits.columns if "loss" in col]
            )
        )
        # print(shouldbe_uniform)
        groups_to_combine = queried_fits.groupby(shouldbe_uniform).ngroups
        if groups_to_combine > 1:
            warnings.warn(
                "the fits that will be aggregated appear to differ by features "
                "other than dataset_name, this may result in unexpected behavior"
            )
        if aggregate_func not in ["mean", "median"]:
            raise ValueError(f"invalid {aggregate_func=} must be mean or median")
        possible_mut_params = set(["shift", "predicted_func_score", "beta"])
        if mut_param not in possible_mut_params:
            raise ValueError(f"invalid {mut_param=}")

        # aggregate mutation values between dataset fits
        muts_df = (
            self.split_apply_combine_muts(
                groupby="dataset_name",
                aggregate_func=aggregate_func,
                inner_merge_dataset_muts=inner_merge_dataset_muts,
                times_seen_threshold=times_seen_threshold,
                phenotype_as_effect=phenotype_as_effect,
                query=query,
            )
            .groupby("mutation")
            .aggregate(aggregate_func)
        )

        # drop columns which are not the mutational parameter of interest
        drop_cols = [c for c in muts_df.columns if "times_seen" in c]
        for param in possible_mut_params - set([mut_param]):
            drop_cols.extend([c for c in muts_df.columns if c.startswith(param)])
        muts_df.drop(drop_cols, axis=1, inplace=True)

        # add in the mutation annotations
        parse_mut = self.fit_models.iloc[0].model.data.parse_mut
        muts_df["wildtype"], muts_df["site"], muts_df["mutant"] = zip(
            *muts_df.reset_index()["mutation"].map(parse_mut)
        )

        # no longer need mutation annotation
        muts_df.reset_index(drop=True, inplace=True)

        wt_dict = {
            "wildtype": self.site_map_union[self.reference].values,
            "mutant": self.site_map_union[self.reference].values,
            "site": self.site_map_union[self.reference].index.values,
        }

        for value_col in [c for c in muts_df.columns if c.startswith(mut_param)]:
            wt_dict[value_col] = 0

        # add reference wildtype values needed for lineplot and heatmap fx
        muts_df = pd.concat([muts_df, pd.DataFrame(wt_dict)])

        # add in wildtype values for each non-reference condition
        # these will be available in the tooltip
        addtl_tooltip_stats = []
        for condition in self.conditions:
            if condition == self.reference:
                continue
            addtl_tooltip_stats.append(f"wildtype_{condition}")
            muts_df[f"wildtype_{condition}"] = muts_df.site.apply(
                lambda site: self.site_map_union.loc[site, condition]
            )

        # melt conditions and stats cols, beta is already "tall"
        # note that we must rename conditions with "." in the
        # name to "_" to avoid altair errors
        # TODO let's just make sure we don't have "." in the condition names
        if mut_param == "beta":
            muts_df_tall = muts_df.assign(condition=self.reference.replace(".", "_"))
        else:
            muts_df_tall = muts_df.melt(
                id_vars=["wildtype", "site", "mutant"] + addtl_tooltip_stats,
                value_vars=[c for c in muts_df.columns if c.startswith(mut_param)],
                var_name="condition",
                value_name=mut_param,
            ).replace(
                {
                    f"{mut_param}_{condition}": condition.replace(".", "_")
                    for condition in self.conditions
                },
            )

        # add in condition colors, rename for altair
        condition_colors = {
            con.replace(".", "_"): col for con, col in self.condition_colors.items()
        }

        # rename for altair
        addtl_tooltip_stats = [v.replace(".", "_") for v in addtl_tooltip_stats]
        muts_df_tall.rename(
            {c: c.replace(".", "_") for c in muts_df_tall.columns}, axis=1, inplace=True
        )

        args = {
            "data_df": muts_df_tall,
            "stat_col": mut_param,
            "addtl_tooltip_stats": addtl_tooltip_stats,
            "category_col": "condition",
            "heatmap_color_scheme": "redblue",
            "init_floor_at_zero": False,
            "categorical_wildtype": True,
            "category_colors": condition_colors,
        }

        return multidms.plot._lineplot_and_heatmap(**args, **kwargs)

    def mut_param_traceplot(
        self,
        mutations,
        mut_param="shift",
        x="scale_coeff_lasso_shift",
        width_scalar=100,
        height_scalar=100,
        **kwargs,
    ):
        """
        visualize mutation parameter values across the lasso penalty weights
        (by default) of a given subset of the mutations in the form of an
        `altair.FacetChart`. This is useful when you would like to confirm
        that a reported mutational parameter value carries through across the
        individual fits.


        Returns
        -------
        altair.Chart
            A chart object which can be displayed in a jupyter notebook
            or saved to a file.
        """
        if isinstance(mutations, str):
            mutations = [mutations]
        if len(mutations) == 0:
            raise ValueError("invalid mutations, must be non-empty list")
        if len(mutations) >= 500:
            raise ValueError("too many mutations, please subset to < 500")
        possible_mut_params = set(["shift", "predicted_func_score", "beta"])
        if mut_param not in possible_mut_params:
            raise ValueError(f"invalid {mut_param=}")

        # get mutation values, group by x axis variable and dataset
        muts_df = self.split_apply_combine_muts(
            groupby=("dataset_name", x), **kwargs
        ).reset_index()

        # drop columns which are not the mutational parameter of interest,
        # or mutational identifiers
        drop_cols = [c for c in muts_df.columns if "times_seen" in c]
        for param in possible_mut_params - set([mut_param]):
            drop_cols.extend([c for c in muts_df.columns if c.startswith(param)])
        muts_df.drop(drop_cols, axis=1, inplace=True)

        # subset to mutations of interest
        muts_df = muts_df.query("mutation.isin(@mutations)")

        # check that we have multiple lasso penalty weights
        if len(muts_df.scale_coeff_lasso_shift.unique()) <= 1:
            raise ValueError(
                "invalid kwargs, must specify a subset of fits with "
                "multiple lasso penalty weights"
            )

        # add in mutation annotations for coloring
        def mut_type(mut):
            return "stop" if mut.endswith("*") else "nonsynonymous"

        muts_df = muts_df.assign(mut_type=muts_df.mutation.apply(mut_type))

        # melt conditions and stats cols, beta is already "tall"
        # id_cols = ["scale_coeff_lasso_shift", "mutation", "is_stop"]
        id_cols = ["dataset_name", x, "mut_type", "mutation"]
        stat_cols_to_keep = [c for c in muts_df.columns if c.startswith(mut_param)]
        if mut_param == "beta":
            muts_df_tall = muts_df.assign(condition=self.reference)
        else:
            muts_df_tall = muts_df.melt(
                id_vars=id_cols,
                value_vars=stat_cols_to_keep,
                var_name="condition",
                value_name=mut_param,
            )
            muts_df_tall.condition = muts_df_tall.condition.str.lstrip(f"{mut_param}_")

        # create altair chart
        highlight = alt.selection_point(
            on="mouseover", fields=["mutation"], nearest=True
        )
        num_facet_rows = len(muts_df_tall.dataset_name.unique())
        num_facet_cols = len(muts_df_tall.condition.unique())

        base = (
            alt.Chart(muts_df_tall)
            .encode(
                x=alt.X(
                    x,
                    type="nominal",
                    title=(
                        PARAMETER_NAMES_FOR_PLOTTING[x]
                        if x in PARAMETER_NAMES_FOR_PLOTTING
                        else x
                    ),
                ),
                y=alt.Y(mut_param, type="quantitative", title=mut_param),
                color="mut_type",
                detail="mutation",
                tooltip=["mutation", mut_param],
            )
            .properties(
                width=num_facet_cols * width_scalar,
                height=num_facet_rows * height_scalar,
            )
        )

        points = base.mark_circle().encode(opacity=alt.value(0)).add_params(highlight)

        lines = base.mark_line().encode(
            size=alt.condition(~highlight, alt.value(1), alt.value(3))
        )

        return alt.layer(points, lines).facet(
            row=alt.Row("dataset_name", title="Replicate"),
            column=alt.Column("condition", title="Experiment"),
        )

    # TODO fix height scalar
    def shift_sparsity(
        self,
        x="scale_coeff_lasso_shift",
        width_scalar=100,
        height_scalar=10,
        return_data=False,
        **kwargs,
    ):
        """
        Visualize shift parameter set sparsity across the lasso penalty weights
        (by default) in the form of an `altair.FacetChart`.
        We will group the mutations according to their status as either a
        a "stop" (e.g. A15*), or "nonsynonymous" (e.g. A15G) mutation before calculating
        the sparsity. This is because in a way, mutations to stop codons act as a
        False positive rate, as we expect their mutational effect to be equally
        deleterious in all experiments, and thus have a shift parameter value of zero.


        Returns
        -------
        altair.Chart or Tuple(pd.DataFrame, altair.Chart)
            A chart object which can be displayed in a jupyter notebook
            or saved to a file. If `return_data=True`, then a tuple
            containing the chart and the underlying data will be returned.
        """
        # get mutation values, group by x axis variable and dataset
        df = self.split_apply_combine_muts(groupby=("dataset_name", x), **kwargs)

        # no need to view parameters besides shifts
        to_throw = [
            col
            for col in df.columns
            if not col.startswith("shift") and col != "mutation"
        ]

        # feature columns for distinct sparsity measurements
        feature_cols = ["dataset_name", x, "mut_type"]

        def sparsity(x):
            return (x == 0).mean()

        def mut_type(mut):
            return "stop" if mut.endswith("*") else "nonsynonymous"

        # apply, drop, and melt
        sparsity_df = (
            df.drop(columns=to_throw)
            .assign(mut_type=lambda x: x.mutation.apply(mut_type))
            .reset_index()
            .groupby(by=feature_cols)
            .apply(sparsity, include_groups=False)
            .drop(columns=["mutation"])
            .reset_index(drop=False)
            .melt(id_vars=feature_cols, var_name="mut_param", value_name="sparsity")
        )
        num_facet_rows = len(sparsity_df.dataset_name.unique())
        num_facet_cols = len(sparsity_df.mut_param.unique())

        # create altair chart
        base_chart = (
            alt.Chart(sparsity_df)
            .encode(
                x=alt.X(
                    "scale_coeff_lasso_shift",
                    type="nominal",
                    title=(
                        PARAMETER_NAMES_FOR_PLOTTING[x]
                        if x in PARAMETER_NAMES_FOR_PLOTTING
                        else x
                    ),
                ).axis(
                    format=".1e",
                ),
                y=alt.Y("sparsity", type="quantitative", title="Sparsity").axis(
                    format="%"
                ),
                color=alt.Color("mut_type", type="nominal", title="Mutation type"),
                tooltip=[
                    "scale_coeff_lasso_shift",
                    "sparsity",
                    "mut_type",
                ],
            )
            .properties(
                width=num_facet_cols * width_scalar,
                height=num_facet_rows * height_scalar,
            )
        )

        # if the x axis is numeric, do line plots, otherwise do bar plots
        if sparsity_df[x].dtype.kind in "biufc":
            chart = base_chart.mark_point() + base_chart.mark_line()
        else:
            chart = base_chart.mark_bar().encode(xOffset="mut_type")

        facetted_chart = chart.facet(
            row=alt.Row("dataset_name", title="Dataset"),
            column=alt.Column("mut_param", title="Experimental Shifts"),
        )

        if return_data:
            return facetted_chart, sparsity_df
        return facetted_chart

    def mut_param_dataset_correlation(
        self,
        x="scale_coeff_lasso_shift",
        width_scalar=150,
        height=200,
        return_data=False,
        r=2,
        **kwargs,
    ):
        """
        Visualize the correlation between replicate datasets across the lasso penalty
        weights (by default) in the form of an `altair.FacetChart`.
        We compute correlation of mutation parameters accross each pair of datasets
        in the collection.

        Parameters
        ----------
        x : str, optional
            The parameter to plot on the x-axis.
            The default is "scale_coeff_lasso_shift".
        width_scalar : int, optional
            The width of the chart. The default is 150.
        height : int, optional
            The height of the chart. The default is 200.
        return_data : bool, optional
            Whether to return the underlying data. The default is False.
        r : int, optional
            The exponential of the correlation coefficient reported.
            May be either 1 for pearson,
            2 for coefficient of determination (r-squared),
            The default is 2.
        **kwargs : dict
            The keyword arguments to pass to the
            :func:`multidms.model_collection.ModelCollection.split_apply_combine_muts`
            method. See the method docstring for details.

        Returns
        -------
        altair.Chart or Tuple(altair.Chart, pd.DataFrame)
            A chart object which can be displayed in a jupyter notebook
            or saved to a file. If `return_data=True`, then a tuple
            containing the chart and the underlying data will be returned.
        """
        if r not in [1, 2]:
            raise ValueError("invalid r, must be 1 or 2")

        query = "dataset_name.notna()" if "query" not in kwargs else kwargs["query"]
        if len(self.fit_models.query(query).dataset_name.unique()) < 2:
            raise ValueError("Must specify a subset of fits with multiple datasets")

        muts_df = self.split_apply_combine_muts(
            groupby=("dataset_name", x), **kwargs
        ).reset_index()

        replicate_series = []
        comparisons = list(it.combinations(muts_df.dataset_name.unique(), 2))
        for datasets in comparisons:
            wide_df = (
                muts_df.query(f"dataset_name.isin({datasets})")
                .drop([c for c in muts_df.columns if "times_seen" in c], axis=1)
                .pivot(columns=["dataset_name", x], index="mutation")
            )
            wide_df.columns.names = ["mut_param"] + wide_df.columns.names[1:]
            for (mut_param, x_i), replicate_params_df in wide_df.T.groupby(
                ["mut_param", x]
            ):
                replicate_series.append(
                    pd.DataFrame(
                        {
                            "datasets": ",".join(datasets),
                            "mut_param": mut_param,
                            "correlation": replicate_params_df.T.corr().iloc[0, 1] ** r,
                            x: x_i,
                        },
                        index=[0],
                    ),
                )

        replicate_df = my_concat(replicate_series)

        title_suffix = "(R^2)" if r == 2 else "(pearson)"
        # create altair chart
        base_chart = (
            alt.Chart(replicate_df)
            .encode(
                x=alt.X(
                    x,
                    type="nominal",
                    title=(
                        PARAMETER_NAMES_FOR_PLOTTING[x]
                        if x in PARAMETER_NAMES_FOR_PLOTTING
                        else x
                    ),
                ).axis(
                    format=".1e",
                ),
                y=alt.Y(
                    "correlation",
                    type="quantitative",
                    title=f"Correlation {title_suffix}",
                ),
                color=alt.Color("mut_param", type="nominal", title="Parameter"),
                tooltip=["datasets", "correlation", "mut_param"],
            )
            .properties(width=len(comparisons) * width_scalar, height=height)
        )

        # if the x axis is numeric, do line plots, otherwise do bar plots
        if replicate_df[x].dtype.kind in "biufc":
            chart = base_chart.mark_point() + base_chart.mark_line()
        else:
            chart = base_chart.mark_bar().encode(xOffset="mut_param")

        facetted_chart = chart.facet(
            column=alt.Column("datasets", title="Experiment comparison"),
        )

        if return_data:
            return facetted_chart, replicate_df
        return facetted_chart
