"""

==========
plot
==========

Plotting functions.

"""


from functools import reduce
import altair as alt
import matplotlib.colors
import natsort
import pandas
import polyclonal.plot
import multidms
from polyclonal.plot import DEFAULT_NEGATIVE_COLOR, DEFAULT_POSITIVE_COLORS



alt.data_transformers.disable_max_rows()



def lineplot_and_heatmap(
    *,
    data_df,
    stat_col,
    category_col,
    alphabet=None,
    sites=None,
    addtl_tooltip_stats=None,
    addtl_slider_stats=None,
    init_floor_at_zero=True,
    init_site_statistic="sum",
    cell_size=11,
    lineplot_width=690,
    lineplot_height=100,
    site_zoom_bar_width=500,
    site_zoom_bar_color_col=None,
    plot_title=None,
    show_single_category_label=False,
    category_colors=None,
    heatmap_negative_color=None,
    heatmap_color_scheme=None,
    heatmap_color_scheme_mid_0=True,
    heatmap_max_at_least=None,
    heatmap_min_at_least=None,
    site_zoom_bar_color_scheme="set3",
    slider_binding_range_kwargs=None,
    category_prefix_as_replicate=False,
    categorical_wildtype=False,
):
    """Lineplots and heatmaps of per-site and per-mutation values.

    Note
    ----
    This code is from polyclonal, I'll be modifying it for use with
    multidms.

    Parameters
    ----------
    data_df : pandas.DataFrame
        Data to plot. Must have columns "site", "wildtype", "mutant", `stat_col`, and
        `category_col`. The wildtype values (wildtype = mutant) should be included,
        but are not used for the slider filtering or included in site summary lineplot.
    stat_col : str
        Column in `data_df` with statistic to plot.
    category_col : str
        Column in `data_df` with category to facet plots over. You can just create
        a dummy column with some dummy value if you only have one category.
    alphabet : array-like or None
        Alphabet letters in order. If `None`, use natsorted "mutant" col of `data_df`.
    sites : array-like or None
        Sites in order. If `None`, use natsorted "site" col of `data_df`.
    addtl_tooltip_stats : None or array-like
        Additional mutation-level stats to show in the heatmap tooltips. Values in
        `addtl_slider_stats` automatically included.
    addtl_slider_stats : None or dict
        Additional stats for which to have a slider, value is initial setting. Ignores
        wildtype and drops it when all mutants have been dropped at site. Null values
        are not filtered.
    init_floor_at_zero : bool
        Initial value for option to put floor of zero on value is `stat_col`.
    init_site_statistic : {'sum', 'mean', 'max', 'min'}
        Initial value for site statistic in lineplot, calculated from `stat_col`.
    cell_size : float
        Size of cells in heatmap
    lineplot_width : float or None
        Overall width of lineplot.
    lineplot_height : float
        Height of line plot.
    site_zoom_bar_width : float
        Width of site zoom bar.
    site_zoom_bar_color_col : float
        Column in `data_df` with which to color zoom bar. Must be the same for all
        entries for a site.
    plot_title : str or None
        Overall plot title.
    show_single_category_label : bool
        Show the category label if just one category.
    category_colors : None or dict
        Map each category to its color, or None to use default. These are the
        colors for **positive** values of `stat_col`.
    heatmap_negative_color : None or str
        Color used for negative values in heatmaps, or None to use default.
    heatmap_color_scheme : None or str
        Heatmap uses this `Vega scheme <https://vega.github.io/vega/docs/schemes>`_
        rather than `category_colors` and `heatmap_negative_color`.
    heatmap_color_scheme_mid_0 : bool
        Set the heatmap color scheme so the domain mid is zero.
    heatmap_max_at_least : None or float
        Make heatmap color max at least this large.
    heatmap_min_at_least : None or float
        Make heatmap color min at least this small, but still set to 0 if floor of zero
        selected.
    site_zoom_bar_color_scheme : str
        If using `site_zoom_bar_color_col`, the
        `Vega color scheme <https://vega.github.io/vega/docs/schemes>`_ to use.
    slider_binding_range_kwargs : dict
        Keyed by keys in ``addtl_slider_stats``, with values being dicts
        giving keyword arguments passed to ``altair.binding_range`` (eg,
        'min', 'max', 'step', etc.
    category_prefix_as_replicate : bool
        The first part of the category will be treated as a key on which to
        merge conditions as if they were replicates. Any number of replicates
        can be matched with any given category. The replicates will be combined
        using an inner merge, and tooltips will be added to each point
        showing the average value of all replicates.
    categorical_wildtype : bool
        if true, then the columns that are prefixed with 'wildtype_' and
        follow with a suffix matching a specific condition, then the heatmaps
        will be labeled with points at all non identical sites when compared
        with the 'wildtype' column (which will be used to mark 'x' on each category
        heatmap). 
    """

    basic_req_cols = ["site", "wildtype", "mutant", stat_col, category_col]

    if addtl_tooltip_stats is None:
        addtl_tooltip_stats = []
    
    # TODO sort tooltips?

    ##############
    # find and merge replicates
    ##############

    


    if addtl_slider_stats is None:
        addtl_slider_stats = {}
    req_cols = basic_req_cols + addtl_tooltip_stats + list(addtl_slider_stats)
    if site_zoom_bar_color_col:
        req_cols.append(site_zoom_bar_color_col)
    req_cols = list(dict.fromkeys(req_cols))  # https://stackoverflow.com/a/17016257
    if not set(req_cols).issubset(data_df.columns):
        raise ValueError(f"Missing required columns\n{data_df.columns=}\n{req_cols=}")
    if any(c.startswith("_stat") for c in req_cols):  # used for calculated stats
        raise ValueError(f"No columns can start with '_stat' in {data_df.columns=}")
    data_df = data_df[req_cols].reset_index(drop=True)

    # filter `data_df` by any minimums in `slider_binding_range_kwargs`
    if slider_binding_range_kwargs is None:
        slider_binding_range_kwargs = {}
    for col, col_kwargs in slider_binding_range_kwargs.items():
        if "min" in col_kwargs:
            data_df = data_df[
                (data_df[col] >= col_kwargs["min"])
                | (data_df["wildtype"] == data_df["mutant"])
            ]

    categories = data_df[category_col].unique().tolist()
    show_category_label = show_single_category_label or (len(categories) > 1)

    # set color schemes if use defaults
    if not category_colors:
        if len(categories) > len(DEFAULT_POSITIVE_COLORS):
            raise ValueError("Explicitly set `category_colors` if this many categories")
        category_colors = dict(zip(categories, DEFAULT_POSITIVE_COLORS))
    if not heatmap_negative_color:
        heatmap_negative_color = DEFAULT_NEGATIVE_COLOR

    no_na_cols = basic_req_cols + (
        [site_zoom_bar_color_col] if site_zoom_bar_color_col else []
    )
    if data_df[no_na_cols].isnull().any().any():
        raise ValueError(
            f"`data_df` has NA values in key cols:\n{data_df[no_na_cols].isnull().any()}"
        )

    if alphabet is None:
        alphabet = natsort.natsorted(data_df["mutant"].unique())
    else:
        data_df = data_df.query("mutant in @alphabet")

    if sites is None:
        sites = natsort.natsorted(data_df["site"].unique(), alg=natsort.ns.SIGNED)
    else:
        data_df = data_df.query("site in @sites")
        sites = [site for site in sites if site in set(data_df["site"])]
    # order sites:
    # https://github.com/dms-vep/dms-vep-pipeline/issues/53#issuecomment-1227817963
    data_df["_stat_site_order"] = data_df["site"].map(
        {site: i for i, site in enumerate(sites)}
    )


    # TODO JARED
    # get tooltips for heatmap
    # For tooltips! this is clever, check out where req_cols
    # comes from ^ line 129
    heatmap_tooltips = [
        alt.Tooltip(c, type="quantitative", format=".3g")
        if data_df[c].dtype == float
        else alt.Tooltip(c, type="nominal")
        for c in req_cols
        if c != category_col or show_category_label
    ]

    # make floor at zero selection, setting floor to either 0 or min in data (no floor)
    min_stat = data_df[stat_col].min()  # used as min in heatmap when not flooring at 0
    if heatmap_min_at_least is not None:
        min_stat = min(min_stat, heatmap_min_at_least)
    max_stat = data_df[stat_col].max()  # used as max in heatmap
    if heatmap_max_at_least is not None:
        max_stat = max(max_stat, heatmap_max_at_least)
    floor_at_zero = alt.selection_point(
        name="floor_at_zero",
        bind=alt.binding_radio(
            options=[0, min_stat],
            labels=["yes", "no"],
            name=f"floor {stat_col} at zero",
        ),
        fields=["floor"],
        value=[{"floor": 0 if init_floor_at_zero else min_stat}],
    )

    # create sliders for max of statistic at site and any additional sliders
    sliders = {}
    for slider_stat, init_slider_stat in addtl_slider_stats.items():
        binding_range_kwargs = {
            "min": data_df[slider_stat].min(),
            "max": data_df[slider_stat].max(),
            "name": f"minimum {slider_stat}",
        }
        if slider_stat in slider_binding_range_kwargs:
            binding_range_kwargs.update(slider_binding_range_kwargs[slider_stat])
        sliders[slider_stat] = alt.selection_point(
            fields=["cutoff"],
            value=[{"cutoff": init_slider_stat}],
            bind=alt.binding_range(**binding_range_kwargs),
        )
    sliders["_stat_site_max"] = alt.selection_point(
        fields=["cutoff"],
        value=[{"cutoff": min_stat}],
        bind=alt.binding_range(
            name=f"minimum max of {stat_col} at site",
            min=min_stat,
            max=max_stat,
        ),
    )

    # whether to show line on line plot
    line_selection = alt.selection_point(
        bind=alt.binding_radio(
            options=[True, False],
            labels=["yes", "no"],
            name="show line on site plot",
        ),
        fields=["_stat_show_line"],
        value=[{"_stat_show_line": True}],
    )

    # create site zoom bar
    site_brush = alt.selection_interval(
        encodings=["x"],
        mark=alt.BrushConfig(stroke="black", strokeWidth=2),
    )
    if site_zoom_bar_color_col:
        site_zoom_bar_df = data_df[
            ["site", "_stat_site_order", site_zoom_bar_color_col]
        ].drop_duplicates()
        if any(site_zoom_bar_df.groupby("site").size() > 1):
            raise ValueError(f"multiple {site_zoom_bar_color_col=} values for sites")
    else:
        site_zoom_bar_df = data_df[["site", "_stat_site_order"]].drop_duplicates()
    site_zoom_bar = (
        alt.Chart(site_zoom_bar_df)
        .mark_rect()
        .encode(
            x=alt.X(
                "site:O",
                sort=alt.EncodingSortField(field="_stat_site_order", order="ascending"),
            ),
            color=(
                alt.Color(
                    site_zoom_bar_color_col,
                    type="nominal",
                    scale=alt.Scale(scheme=site_zoom_bar_color_scheme),
                    legend=alt.Legend(orient="left"),
                    sort=(
                        site_zoom_bar_df.set_index("site")
                        .loc[sites][site_zoom_bar_color_col]
                        .unique()
                    ),
                )
                if site_zoom_bar_color_col
                else alt.value("gray")
            ),
            tooltip=[c for c in site_zoom_bar_df.columns if not c.startswith("_stat")],
        )
        .mark_rect()
        .add_parameter(site_brush)
        .properties(width=site_zoom_bar_width, height=cell_size, title="site zoom bar")
    )

    # to make data in Chart smaller, access properties that are same across all sites
    # or categories via a transform_lookup. Make data frames with columns to do that.
    lookup_dfs = {}
    for lookup_col in ["site", category_col]:
        cols_to_lookup = [
            c
            for c in data_df.columns
            if all(data_df.groupby(lookup_col)[c].nunique(dropna=False) == 1)
            if c not in ["site", category_col]
        ]
        if cols_to_lookup:
            lookup_dfs[lookup_col] = data_df[
                [lookup_col, *cols_to_lookup]
            ].drop_duplicates()
            assert len(lookup_dfs[lookup_col]) == data_df[lookup_col].nunique()
            data_df = data_df.drop(columns=cols_to_lookup)

    ### should we make another base chart for wildtypes to be added to ?

    # make the base chart that holds the data and common elements
    base_chart = alt.Chart(data_df)
    for lookup_col, lookup_df in lookup_dfs.items():
        base_chart = base_chart.transform_lookup(
            lookup=lookup_col,
            from_=alt.LookupData(
                data=lookup_df,
                key=lookup_col,
                fields=[c for c in lookup_df.columns if c != lookup_col],
            ),
        )

    # Transforms on base chart. The "_stat" columns is floor transformed stat_col.
    base_chart = base_chart.transform_calculate(
        _stat=alt.expr.max(alt.datum[stat_col], floor_at_zero["floor"]),
    )

    # Filter data using slider stat
    assert list(sliders)[-1] == "_stat_site_max"  # last for right operation order
    for slider_stat, slider in sliders.items():
        if slider_stat == "_stat_site_max":
            base_chart = base_chart.transform_joinaggregate(
                _stat_site_max="max(_stat)",
                groupby=["site"],
            )
        base_chart = base_chart.transform_filter(
            (alt.datum[slider_stat] >= slider["cutoff"] - 1e-6)  # add rounding tol
            | ~alt.expr.isNumber(alt.datum[slider_stat])  # do not filter null values
        )
    # Remove any sites that are only wildtype and filter with site zoom brush
    base_chart = (
        base_chart.transform_calculate(
            _stat_not_wildtype=alt.datum.wildtype != alt.datum.mutant
        )
        .transform_joinaggregate(
            _stat_site_has_non_wildtype="max(_stat_not_wildtype)",
            groupby=["site"],
        )
        .transform_filter(alt.datum["_stat_site_has_non_wildtype"])
        .transform_filter(site_brush)
    )

    ### 

    # make the site chart
    site_statistics = ["sum", "mean", "max", "min"]
    if init_site_statistic not in site_statistics:
        raise ValueError(f"invalid {init_site_statistic=}")
    if set(site_statistics).intersection(req_cols):
        raise ValueError(f"`data_df` cannot have these columns:\n{site_statistics}")
    site_stat = alt.selection_point(
        bind=alt.binding_radio(
            labels=site_statistics,
            options=[f"_stat_{stat}" for stat in site_statistics],
            name=f"site {stat_col} statistic",
        ),
        fields=["_stat_site_stat"],
        value=[{"_stat_site_stat": f"_stat_{init_site_statistic}"}],
        name="site_stat",
    )
    site_prop_cols = lookup_dfs["site"].columns if "site" in lookup_dfs else ["site"]

    lineplot_base = (
        #base_chart.transform_filter(alt.datum.wildtype != alt.datum.mutant)
        base_chart
        .transform_aggregate(
            **{f"_stat_{stat}": f"{stat}(_stat)" for stat in site_statistics},
            groupby=[*site_prop_cols, category_col],
        )
        .transform_fold(
            [f"_stat_{stat}" for stat in site_statistics],
            ["_stat_site_stat", "_stat_site_val"],
        )
        .transform_filter(site_stat)
        .encode(
            x=alt.X(
                "site:O",
                sort=alt.EncodingSortField(field="_stat_site_order", order="ascending"),
            ),
            y=alt.Y(
                "_stat_site_val:Q",
                scale=alt.Scale(zero=True),
                title=f"site {stat_col}",
            ),
            color=alt.Color(
                category_col,
                scale=alt.Scale(
                    domain=categories,
                    range=[category_colors[c] for c in categories],
                ),
                legend=alt.Legend(orient="left") if show_category_label else None,
            ),
            tooltip=[
                "site",
                *([category_col] if show_category_label else []),
                alt.Tooltip("_stat_site_val:Q", format=".3g", title=f"site {stat_col}"),
                *[
                    f"{c}:N"
                    for c in site_prop_cols
                    if c != "site" and not c.startswith("_stat")
                ],
            ],
        )
    )

    site_lineplot = (
        (
            (
                lineplot_base.mark_line(size=1, opacity=0.7)
                .transform_calculate(_stat_show_line="true")
                .transform_filter(line_selection)
            )
            + lineplot_base.mark_circle(opacity=0.7)
        )
        .add_parameter(site_stat, line_selection)
        .properties(width=lineplot_width, height=lineplot_height)
    )

    # make base chart for heatmaps

    heatmap_base = base_chart.encode(
        y=alt.Y(
            "mutant",
            sort=alphabet,
            scale=alt.Scale(domain=alphabet),
            title=None,
        ),
    )

    # wildtype text marks for heatmap
    #if reference_category is not None:
    #if categorical_wildtype:

    wildtype = heatmap_base.transform_filter(alt.datum.mutant == alt.datum.wildtype).encode(
        x=alt.X(
            "site:O",
            sort=alt.EncodingSortField(field="_stat_site_order", order="ascending"),
        ),
    ).transform_filter(
        alt.datum.wildtype == alt.datum.mutant
    ).mark_text(text="x", color="black")

    #else:
    #    reference_wildtype = 

    #    heatmap_base.encode(
    #        x=alt.X(
    #            "site:O",
    #            sort=alt.EncodingSortField(field="_stat_site_order", order="ascending"),
    #        ),
    #    ).transform_filter(
    #        alt.datum.wildtype == alt.datum.mutant
    #    ).mark_text(text="x", color="black")

    # background fill for missing values in heatmap, imputing dummy stat
    # to get all cells
    #heatmap_bg = (
    #    heatmap_base.encode(
    #        x=alt.X(
    #            "site:O",
    #            sort=alt.EncodingSortField(field="_stat_site_order", order="ascending"),
    #        )
    #    )
    #    .transform_impute(
    #        impute="_stat_dummy",
    #        key="mutant",
    #        keyvals=alphabet,
    #        groupby=["site"],
    #        value=None,
    #    )
    #    .mark_rect(color="gray", opacity=0.25)
    #)

    heatmaps = []

    # Make heatmaps for each category and vertically concatenate. We do this in loop
    # rather than faceting to enable compound chart w wildtype marks and category
    # specific coloring.
    for category in categories:

        background = heatmap_base.transform_filter(alt.datum[category_col] == category).encode(
                x=alt.X(
                    "site:O",
                    sort=alt.EncodingSortField(field="_stat_site_order", order="ascending"),
                )
            ).transform_impute(
                impute="_stat_dummy",
                key="mutant",
                keyvals=alphabet,
                groupby=["site"],
                value=None,
            ).mark_rect(color="gray", opacity=0.25)

        data = heatmap_base.transform_filter(alt.datum[category_col] == category).encode(
                x=alt.X(
                    "site:O",
                    sort=alt.EncodingSortField(
                        field="_stat_site_order",
                        order="ascending",
                    ),
                    # only show ticks and axis title on bottom most category
                    axis=alt.Axis(
                        labels=category == categories[-1],
                        ticks=category == categories[-1],
                        title="site" if category == categories[-1] else None,
                    ),
                ),
                color=alt.Color(
                    "_stat:Q",
                    legend=alt.Legend(
                        orient="left",
                        title=stat_col,
                        titleOrient="left",
                        gradientLength=100,
                        gradientStrokeColor="black",
                        gradientStrokeWidth=0.5,
                    ),
                    scale=alt.Scale(
                        domainMax=max_stat,
                        domainMin=alt.ExprRef("floor_at_zero.floor"),
                        zero=True,
                        nice=False,
                        type="linear",
                        **({"domainMid": 0} if heatmap_color_scheme_mid_0 else {}),
                        **(
                            {"scheme": heatmap_color_scheme}
                            if heatmap_color_scheme
                            else {
                                "range": (
                                    color_gradient_hex(
                                        heatmap_negative_color, "white", n=20
                                    )
                                    + color_gradient_hex(
                                        "white", category_colors[category], n=20
                                    )[1:]
                                )
                            }
                        ),
                    ),
                ),
                stroke=alt.value("black"),
                tooltip=heatmap_tooltips,
            ).mark_rect().properties(
                width=alt.Step(cell_size),
                height=alt.Step(cell_size),
                title=alt.TitleParams(
                    category if show_category_label else "",
                    color=category_colors[category],
                    anchor="middle",
                    orient="left",
                ),
            )
        heatmap = background + data + wildtype
        #print(addtl_tooltip_stats)
        #print(f"wildtype_{category}")
        #if reference_category is not None:
        if categorical_wildtype and f"wildtype_{category}" in addtl_tooltip_stats:
            #print("yo")

            heatmap += heatmap_base.transform_filter(
                    alt.datum[category_col] == category
                ).encode(
                    x=alt.X(
                        "site:O",
                        sort=alt.EncodingSortField(field="_stat_site_order", order="ascending"),
                    ),
                ).transform_filter(
                    alt.datum[f"wildtype_{category}"] != alt.datum.wildtype
                ).transform_filter(
                    alt.datum[f"wildtype_{category}"] == alt.datum.mutant
                ).mark_point(fill=category_colors[category])

        #else:
        #    wildtype = heatmap_base.encode(
        #            x=alt.X(
        #                "site:O",
        #                sort=alt.EncodingSortField(field="_stat_site_order", order="ascending"),
        #            ),
        #        ).transform_filter(
        #            alt.datum.wildtype == alt.datum.mutant
        #        ).mark_text(text="x", color="black")

        heatmaps.append(heatmap)

    heatmaps = alt.vconcat(
        *heatmaps,
        spacing=10,
    ).resolve_scale(
        x="shared",
        color="shared"
        if heatmap_color_scheme or len(categories) == 1
        else "independent",
    )

    chart = (
        alt.vconcat(site_zoom_bar, site_lineplot, heatmaps)
        .add_parameter(floor_at_zero, site_brush, *sliders.values())
        .configure(padding=10)
        .configure_axis(labelOverlap="parity", grid=False)
        .resolve_scale(color="independent")
    )

    if plot_title:
        chart = chart.properties(
            title=alt.TitleParams(
                plot_title,
                anchor="start",
                align="left",
                fontSize=16,
            ),
        )

    return chart


def mut_shift_plot(
    fit_data,
    biochem_order_aas=True,
    times_seen_threshold=3,
    include_beta=True,
    # include_times_seen
    **kwargs,
):
    """Make plot of mutation escape values for one or more replicate fits.

    Parameters
    ----------
    biochem_order_aas : bool
        Biochemically order amino-acid alphabet :attr:`PolyclonalCollection.alphabet`
        by passing it through :func:`polyclonal.alphabets.biochem_order_aas`.
    times_seen_threshold : int
        Set a threshold for the number of genetic backgrounds each mutant must be seen
        within each condition in order to be used in the visualization.
    include_beta : bool
        If True, include the beta values as another category in the figure.
        If False, only include beta's in the tooltip.
    **kwargs
        Keyword args for :func:`polyclonal.plot.lineplot_and_heatmap`
    Returns
    -------
    altair.Chart
        Interactive heat maps.
    """


    kwargs["addtl_tooltip_stats"] = []
    id_vars=["wildtype", "site", "mutant"]

    if type(fit_data) == multidms.model.MultiDmsModel:
        
        mut_df = fit_data.mutations_df        
        times_seen_cols = [c for c in mut_df.columns if "times" in c]
        for c in times_seen_cols:
            mut_df = mut_df[mut_df[c] >= times_seen_threshold]
        fit = fit_data

    elif type(fit_data) == dict:
        if len(fit_data) <= 1:
            raise ValueError("If passing a dictionary of fits, there must be more that one")

        # TODO check if the fits are comparible. 
        # well as long as they're the same then we can pretty much 
        # use the latest fit object in the iteration to get what we need
        # 1. Check that the conditions and their parameters match
        # 2. Make sure their inferred site maps are also subsets of each other 
        # 3. Could also compare the condition colors, yea?
        
        # obtain and curate each of the replicate mutational dataframes
        mutations_dfs = []
        for replicate, fit in fit_data.items():
            
            fit_mut_df = fit.mutations_df.set_index("mutation")
            
            new_column_name_map = {c:f"{replicate}_{c}" for c in fit_mut_df.columns}
            fit_mut_df = fit_mut_df.rename(new_column_name_map, axis=1)

            times_seen_cols = [c for c in fit_mut_df.columns if "times" in c]
            for c in times_seen_cols:
                fit_mut_df = fit_mut_df[fit_mut_df[c] >= times_seen_threshold]
            # TODO these could potentially be slider stats ... ?
            # you would need to change the rep1 replicate for this to work
            # with the wide_to_long call below
            fit_mut_df.drop(times_seen_cols, axis=1, inplace=True)
            mutations_dfs.append(fit_mut_df.assign(replicate=replicate))

        # return pandas.concat(mutations_dfs)
        # no need to do the below
        
        # merge each of the replicate mutational dataframes
        mut_df = reduce(
            lambda  left,right: pandas.merge(
                left,
                right,
                left_index=True, 
                right_index=True,
                how='inner'
            ), 
            mutations_dfs
        )
        #print(mut_df)
        #return None

        
        ###############
        # put the rest of this in lineplot_and_heatmap()?


        # now compute replicate averages
        for c in fit.mutations_df.columns:
            if c == "mutation" or "times_seen" in c: continue
            cols_to_combine = [f"{replicate}_{c}" for replicate in fit_data.keys()]
            if c in ["wts", "sites", "muts"]:
                mut_df[c] = mut_df[cols_to_combine[0]]
                mut_df.drop(cols_to_combine, axis=1, inplace=True)
            else:
                #print(cols_to_combine)
                mut_df[f"value_{c}"] = mut_df[cols_to_combine].mean(axis=1)
                #kwargs["addtl_tooltip_stats"].extend(cols_to_combine)
                #id_vars.extend(cols_to_combine)

    else:
        raise ValueError("'fit_data' must be o type `multidms.MultiDmsModel` or dict")

    # colors must be hex
    condition_colors = {
        f"S_{con}".replace(".", "_"): matplotlib.colors.rgb2hex(tuple(col))
        for con, col in fit.data.condition_colors.items()
        if con != fit.data.reference
    }
    kwargs["category_colors"] = condition_colors

    # All non reference shift parameters are the primary source of data
    value_vars = [
        f"S_{c}".replace(".", "_")
        for c in fit.data.conditions
        if c != fit.data.reference
    ]

    # Beta can either be a tooltip, or a value condition
    if include_beta:
        condition_colors["β"] = matplotlib.colors.rgb2hex((0.,0.,0.))
        value_vars.insert(0, "β")
    else:
        mut_df = mut_df.rename({"value_β": "β"}, axis=1)
        id_vars.insert(0, "β")
        for rep_replicate in list(fit_data.keys()):
            mut_df = mut_df.rename({f"{rep_replicate}_β": f"β ({rep_replicate})"}, axis=1)
            id_vars.insert(1, f"β ({rep_replicate})")

    mut_df = (
        mut_df.rename({c: c.replace(".", "_") for c in mut_df.columns}, axis=1)
        .rename({"wts": "wildtype", "sites": "site", "muts": "mutant"}, axis=1)
    )

    # stack all condition, but leave replicate values in their own column
    mut_df = pandas.wide_to_long(
        mut_df, 
        stubnames=list(fit_data.keys())+["value"], 
        i=id_vars,
        j="condition", 
        sep="_", 
        suffix=".+"        
    ).reset_index()

    for condition in fit.data.conditions:
        if condition == fit.data.reference: continue
        mut_df[f"wildtype_S_{condition}"] = mut_df["wildtype"]
        cond_non_iden_sites = fit.data.non_identical_sites[condition]
        for idx, nis in cond_non_iden_sites.iterrows():
            #nis_idx = mut_df.query(f"site == {idx} & condition == 'S_{condition}'").index
            nis_idx = mut_df.query(f"site == {idx}").index
            mut_df.loc[nis_idx, f"wildtype_S_{condition}"] = nis[condition]
        kwargs["addtl_tooltip_stats"].append(f"wildtype_S_{condition}")

    # print(mut_df.query("site == 19"))

    #mut_df = (
    #    mut_df.rename({c: c.replace(".", "_") for c in mut_df.columns}, axis=1)
    #    .rename({"wts": "wildtype", "sites": "site", "muts": "mutant"}, axis=1)
    #    .melt(
    #        id_vars=id_vars,
    #        value_vars=value_vars,
    #        var_name="condition",
    #    )
    #)

    #print(f"melted mut_df:\n{mut_df}")
    #return

    # TODO we'll need to define the site map as the inter, I think?
    # but maybe not because the inner merge should guerentee that we
    # add all necesary wildtypes, and a few extra that get thrown out
    # by the altair code above.

    category_wt = f"S_{condition}".replace(".", "_") if not include_beta else "β"
    reference_wts = fit.data.site_map[fit.data.reference]
    con_wt_dict = {
            "wildtype": reference_wts.values,
            "mutant": reference_wts.values,
            "site": reference_wts.index.values,
            "value": 0,
            "condition": category_wt
    }
    for rep_replicate in list(fit_data.keys()):
        con_wt_dict[rep_replicate] = 0 
    con_wt = pandas.DataFrame(con_wt_dict)
    mut_df = pandas.concat([con_wt, mut_df])

    #sm = fit.data.site_map[fit.data.reference].copy()
    #sm.index.name = "site"
    #sm = sm.reset_index().rename({fit.data.reference:"wildtype"}, axis=1)
    #kwargs["reference_wildtype"] = sm

    #for condition, wts in fit.data.site_map.items():
    #    
    #    is_ref = condition == fit.data.reference
    #    if is_ref and not include_beta: continue
    #    category_wt = f"S_{condition}".replace(".", "_") if not is_ref else "β"
    #    con_wt_dict = {
    #            "wildtype": wts.values,
    #            "mutant": wts.values,
    #            "site": wts.index.values,
    #            "value": 0,
    #            "condition": category_wt
    #    }
    #    for rep_prefix in list(fit_data.keys()):
    #        con_wt_dict[rep_prefix] = 0 
    #    con_wt = pandas.DataFrame(con_wt_dict)
    #    mut_df = pandas.concat([mut_df, con_wt])

    #break

        # here, we'll want to drop the non identical sites.
        # con_wt.drop()
    

    kwargs["data_df"] = mut_df
    kwargs["stat_col"] = "value"
    kwargs["category_col"] = "condition"
    kwargs["heatmap_color_scheme"] = "redblue"
    kwargs["init_floor_at_zero"] = False
    kwargs["categorical_wildtype"] = True

    for rep_replicate in list(fit_data.keys()):
        kwargs["addtl_tooltip_stats"].append(rep_replicate)

    if "alphabet" not in kwargs:
        kwargs["alphabet"] = fit.data.alphabet

    if biochem_order_aas:
        kwargs["alphabet"] = polyclonal.alphabets.biochem_order_aas(
            kwargs["alphabet"]
        )

    return lineplot_and_heatmap(**kwargs)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
