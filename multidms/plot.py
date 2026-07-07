"""
==========
plot
==========

All interactive Altair-based visualizations for ``multidms``.

Every public function accepts DataFrames and returns ``alt.Chart`` objects,
making them independently testable and usable with any data source.

Public functions
----------------
- :func:`color_gradient_hex` — linear color gradient utility
- :func:`lineplot_and_heatmap` — interactive per-site/per-mutation heatmap
- :func:`convergence_trajectory` — convergence diagnostics with group dropdown
- :func:`mut_param_heatmap` — mutation parameter heatmap (wraps lineplot_and_heatmap)
- :func:`mut_param_traceplot` — mutation parameters across regularization
- :func:`shift_sparsity` — shift sparsity across regularization
- :func:`mut_param_dataset_correlation` — replicate correlation
- :func:`replicate_param_scatter` — replicate parameter scatter with identity line
- :func:`times_seen_hist` — mutation occurrence histogram
- :func:`func_score_boxplot` — functional score distribution
- :func:`ge_landscape` — global epistasis landscape
"""


import altair as alt
import matplotlib.colors
import natsort
import pandas as pd
from scipy import stats

# Colorblind-friendly palette (formerly from polyclonal.plot)
DEFAULT_POSITIVE_COLORS = ("#0072B2", "#CC79A7", "#009E73", "#17BECF", "#BCDB22")
DEFAULT_NEGATIVE_COLOR = "#E69F00"


alt.data_transformers.disable_max_rows()


def color_gradient_hex(start, end, n):
    """Get a list of colors linearly spanning a range.

    Parameters
    ----------
    start : str
        Starting color.
    end : str
        Ending color.
    n : int
        Number of colors in list.

    Returns
    -------
    list
        List of hex codes for colors spanning `start` to `end`.

    Example
    -------
    >>> import multidms.plot as mplt
    >>> mplt.color_gradient_hex('white', 'red', n=5)
    ['#ffffff', '#ffbfbf', '#ff8080', '#ff4040', '#ff0000']

    """
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        name="_", colors=[start, end], N=n
    )
    return [matplotlib.colors.rgb2hex(tup) for tup in cmap(list(range(0, n)))]


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
    show_single_category_label=True,
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
    This code is _mostly_ from ``polyclonal``, but modified for use with ``multidms``

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
        if true, then the columns that are prefixed with 'wildtype' and
        follow with a suffix matching a specific condition,
        will be labeled with points at all non identical sites when compared
        with the 'wildtype' column (which will be used to mark 'x' on each category
        heatmap).
    """
    basic_req_cols = ["site", "wildtype", "mutant", stat_col, category_col]

    if addtl_tooltip_stats is None:
        addtl_tooltip_stats = []

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
        .add_params(site_brush)
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
        base_chart.transform_aggregate(
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
        .add_params(site_stat, line_selection)
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

    wildtype = (
        heatmap_base.transform_filter(alt.datum.mutant == alt.datum.wildtype)
        .encode(
            x=alt.X(
                "site:O",
                sort=alt.EncodingSortField(field="_stat_site_order", order="ascending"),
            ),
        )
        .transform_filter(alt.datum.wildtype == alt.datum.mutant)
        .mark_text(text="x", color="black")
    )

    heatmaps = []

    # Make heatmaps for each category and vertically concatenate. We do this in loop
    # rather than faceting to enable compound chart w wildtype marks and category
    # specific coloring.
    for category in categories:
        background = (
            heatmap_base.transform_filter(alt.datum[category_col] == category)
            .encode(
                x=alt.X(
                    "site:O",
                    sort=alt.EncodingSortField(
                        field="_stat_site_order", order="ascending"
                    ),
                )
            )
            .transform_impute(
                impute="_stat_dummy",
                key="mutant",
                keyvals=alphabet,
                groupby=["site"],
                value=None,
            )
            .mark_rect(color="gray", opacity=0.25)
        )

        data = (
            heatmap_base.transform_filter(alt.datum[category_col] == category)
            .encode(
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
            )
            .mark_rect()
            .properties(
                width=alt.Step(cell_size),
                height=alt.Step(cell_size),
                title=alt.TitleParams(
                    category if show_category_label else "",
                    color=category_colors[category],
                    anchor="middle",
                    orient="left",
                ),
            )
        )
        heatmap = background + data + wildtype
        if categorical_wildtype and f"wildtype_{category}" in addtl_tooltip_stats:
            heatmap += (
                heatmap_base.transform_filter(alt.datum[category_col] == category)
                .encode(
                    x=alt.X(
                        "site:O",
                        sort=alt.EncodingSortField(
                            field="_stat_site_order", order="ascending"
                        ),
                    ),
                )
                .transform_filter(
                    alt.datum[f"wildtype_{category}"] != alt.datum.wildtype
                )
                .transform_filter(alt.datum[f"wildtype_{category}"] == alt.datum.mutant)
                .mark_point(fill=category_colors[category])
            )

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
        .add_params(floor_at_zero, site_brush, *sliders.values())
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


# Per-condition parameter groups that should always use a linear y-axis.
# These contain real-valued parameters (possibly zero or negative) that are
# undefined on a log scale.
LINEAR_SCALE_GROUPS = {"beta0_condition", "alpha", "theta", "sparsity"}

CONVERGENCE_TRAJECTORY_GROUPS = {
    "loss": ["loss_trajectory"],
    "loss_per_variant": ["loss_per_variant_trajectory"],
    "objective_total": ["objective_total_trajectory"],
    "objective_error": ["objective_error_trajectory"],
    "alpha": ["alpha"],
    "block_errors": [
        "calibration_error",
        "beta0_error",
        "beta_nonbundle_error",
        "beta_bundle_error",
    ],
    "block_stepsizes": [
        "calibration_stepsize",
        "beta0_stepsize",
        "beta_nonbundle_stepsize",
        "beta_bundle_stepsize",
    ],
    "block_iterations": [
        "calibration_iter_num",
        "beta0_iter_num",
        "beta_nonbundle_iter_num",
        "beta_bundle_iter_num",
    ],
}


def _detect_per_condition_groups(columns):
    """Detect per-condition column groups from DataFrame columns.

    Looks for columns matching ``loss_{cond}``, ``loss_per_variant_{cond}``,
    ``alpha_{cond}``, ``theta_{cond}``, ``beta0_{cond}``, and
    ``sparsity_{cond}`` patterns and groups them by parameter type.

    Prefixes are processed in decreasing length order to avoid ambiguity
    (e.g., ``loss_per_variant_Delta`` matching both ``loss_`` and
    ``loss_per_variant_``).

    Parameters
    ----------
    columns : list of str
        DataFrame column names.

    Returns
    -------
    dict
        Mapping from group name to list of matching column names.
    """
    prefixes = [
        ("loss_per_variant_per_condition", "loss_per_variant_"),
        ("loss_per_condition", "loss_"),
        ("alpha", "alpha_"),
        ("theta", "theta_"),
        ("beta0_condition", "beta0_"),
        ("sparsity", "sparsity_"),
    ]
    # These are the base (non-condition) columns that start with the same prefix
    base_cols = set(CONVERGENCE_TRAJECTORY_GROUPS.get("overall", []))
    for group_cols in CONVERGENCE_TRAJECTORY_GROUPS.values():
        base_cols.update(group_cols)

    groups = {}
    assigned = set()
    for group_name, prefix in sorted(prefixes, key=lambda x: -len(x[1])):
        matching = [
            c
            for c in columns
            if c.startswith(prefix) and c not in base_cols and c not in assigned
        ]
        if matching:
            groups[group_name] = sorted(matching)
            assigned.update(matching)
    return groups


def convergence_trajectory(
    df,
    *,
    x="iteration",
    id_cols=None,
    tooltip_cols=None,
    trajectory_groups=None,
    init_group="loss",
    log_y=True,
    skip_first=True,
    width=700,
    height=250,
    title="Convergence Diagnostics",
):
    """Interactive convergence trajectory plot.

    Creates an Altair chart with a dropdown to switch between groups
    of convergence diagnostics (overall loss, block errors, block
    stepsizes, block iterations, per-condition parameters). Within
    each group, individual metrics can be toggled via legend clicks.

    Parameters
    ----------
    df : pandas.DataFrame
        Convergence trajectory data, typically from
        ``ModelCollection.convergence_trajectory_df()`` or
        ``Model.convergence_trajectory_df``. Must contain an
        ``iteration`` column and one or more trajectory value columns.
    x : str
        Column for the x-axis. Default ``'iteration'``.
    id_cols : list of str or None
        Columns that identify distinct model runs (e.g.
        ``['dataset_name', 'fusionreg']``). Each unique combination
        gets a distinct line style. If None, no model identity
        distinction is made.
    tooltip_cols : list of str or None
        Extra columns to carry into the hover tooltip (e.g.
        ``['dataset_name', 'fusionreg']``). Any name present in ``df`` and
        not already in ``id_cols`` is preserved through the melt and shown
        on hover. These do NOT contribute to line identity (``model_id``);
        grouping is determined by ``id_cols`` alone.
    trajectory_groups : dict or None
        Mapping from group name to list of column names. If None,
        auto-detected from the DataFrame using canonical groupings
        plus any per-condition columns.
    init_group : str
        Which group to display initially. Default ``'overall'``.
    log_y : bool
        Whether to use log scale on the y-axis. Default True.
    skip_first : bool
        If True, filter out ``iteration == 0``. Default True.
    width : int
        Chart width in pixels. Default 700.
    height : int
        Chart height in pixels. Default 250.
    title : str or None
        Chart title. Default ``'Convergence Diagnostics'``.

    Returns
    -------
    alt.Chart
        Interactive Altair chart with group dropdown and legend toggle.
    """
    df = df.copy()

    if skip_first:
        df = df[df[x] > 0]

    # Build trajectory groups
    if trajectory_groups is None:
        trajectory_groups = {}
        for group_name, group_cols in CONVERGENCE_TRAJECTORY_GROUPS.items():
            present = [c for c in group_cols if c in df.columns]
            if present:
                trajectory_groups[group_name] = present
        # Add per-condition groups
        trajectory_groups.update(_detect_per_condition_groups(df.columns))

    if not trajectory_groups:
        raise ValueError("No trajectory columns found in DataFrame.")

    if init_group not in trajectory_groups:
        init_group = next(iter(trajectory_groups))

    # Collect all value columns
    value_cols = [col for cols in trajectory_groups.values() for col in cols]
    # Build metric -> group mapping
    metric_to_group = {
        col: group for group, cols in trajectory_groups.items() for col in cols
    }

    # Build id columns
    if id_cols is None:
        id_cols = []

    # Extra columns to preserve for tooltips only (not for line identity).
    if tooltip_cols is None:
        tooltip_cols = []
    extra_tooltip_cols = [
        c for c in tooltip_cols if c in df.columns and c not in id_cols
    ]

    melt_id_vars = [x] + id_cols + extra_tooltip_cols
    keep_cols = melt_id_vars + value_cols
    keep_cols = [c for c in keep_cols if c in df.columns]

    # Melt to long form
    long_df = df[keep_cols].melt(
        id_vars=[c for c in melt_id_vars if c in df.columns],
        value_vars=[c for c in value_cols if c in df.columns],
        var_name="metric",
        value_name="value",
    )
    long_df["group"] = long_df["metric"].map(metric_to_group)

    # Build model_id from id_cols
    if id_cols:
        long_df["model_id"] = long_df[id_cols].astype(str).agg(" | ".join, axis=1)
    else:
        long_df["model_id"] = "model"

    # Group dropdown
    group_options = list(trajectory_groups.keys())
    group_selector = alt.selection_point(
        fields=["group"],
        bind=alt.binding_select(
            options=group_options,
            name="Trajectory group ",
        ),
        value=[{"group": init_group}],
    )

    # Legend toggles
    metric_toggle = alt.selection_point(
        fields=["metric"],
        bind="legend",
    )
    model_toggle = alt.selection_point(
        fields=["model_id"],
        bind="legend",
    )

    tooltip_fields = [
        alt.Tooltip(f"{x}:Q"),
        alt.Tooltip("metric:N"),
        alt.Tooltip("value:Q", format=".4g"),
    ]
    if id_cols:
        tooltip_fields.append(alt.Tooltip("model_id:N", title="model"))
    for col in extra_tooltip_cols:
        tooltip_fields.append(alt.Tooltip(f"{col}:N"))

    base = alt.Chart(long_df).transform_filter(group_selector)

    shared_encoding = dict(
        x=alt.X(f"{x}:Q", title="Iteration"),
        color=alt.Color("metric:N", title="Metric"),
        strokeDash=alt.StrokeDash("model_id:N", title="Model"),
        opacity=alt.condition(
            metric_toggle & model_toggle, alt.value(1), alt.value(0.1)
        ),
        tooltip=tooltip_fields,
    )

    if log_y:
        # Determine which groups need a linear y-axis (parameter values
        # that can be zero or negative are undefined on a log scale).
        linear_groups = set(LINEAR_SCALE_GROUPS)
        for group_name in trajectory_groups:
            group_vals = long_df[long_df["group"] == group_name]["value"]
            if (group_vals <= 0).any():
                linear_groups.add(group_name)

        long_df = long_df.assign(
            scale_type=long_df["group"].apply(
                lambda g: "linear" if g in linear_groups else "log"
            )
        )

        # Build two layers: log-scale for loss/error groups, linear for params
        base = alt.Chart(long_df).transform_filter(group_selector)
        log_layer = (
            base.transform_filter(alt.datum.scale_type == "log")
            .mark_line()
            .encode(
                y=alt.Y("value:Q", title="Value", scale=alt.Scale(type="log")),
                **shared_encoding,
            )
        )
        linear_layer = (
            base.transform_filter(alt.datum.scale_type == "linear")
            .mark_line()
            .encode(
                y=alt.Y("value:Q", title="Value", scale=alt.Scale()),
                **shared_encoding,
            )
        )
        chart = (
            (log_layer + linear_layer)
            .resolve_scale(y="independent")
            .add_params(group_selector, metric_toggle, model_toggle)
        )
    else:
        lines = base.mark_line().encode(
            y=alt.Y("value:Q", title="Value", scale=alt.Scale()),
            **shared_encoding,
        )
        chart = lines.add_params(group_selector, metric_toggle, model_toggle)

    chart = chart.properties(width=width, height=height)

    if title:
        chart = chart.properties(
            title=alt.TitleParams(title, anchor="start", fontSize=14)
        )

    return chart


PARAMETER_NAMES_FOR_PLOTTING = {
    "fusionreg": "Fusion Regularization",
}


def mut_param_heatmap(muts_df_tall, *, mut_param="shift", **kwargs):
    """Interactive heatmap of mutation parameters across conditions.

    Thin wrapper around :func:`lineplot_and_heatmap` with mutation-specific
    defaults.

    Parameters
    ----------
    muts_df_tall : pandas.DataFrame
        Long-form DataFrame with columns ``site``, ``wildtype``,
        ``mutant``, ``condition``, and ``mut_param``.
    mut_param : str
        Name of the statistic column in ``muts_df_tall``.
    **kwargs
        Passed to :func:`lineplot_and_heatmap`.

    Returns
    -------
    alt.Chart
    """
    defaults = {
        "data_df": muts_df_tall,
        "stat_col": mut_param,
        "category_col": "condition",
        "heatmap_color_scheme": "redblue",
        "init_floor_at_zero": False,
        "categorical_wildtype": True,
    }
    defaults.update(kwargs)
    return lineplot_and_heatmap(**defaults)


def mut_param_traceplot(
    muts_df_tall,
    *,
    x="fusionreg",
    mut_param="shift",
    width_scalar=100,
    height_scalar=100,
):
    """Trace mutation parameter values across regularization weights.

    Parameters
    ----------
    muts_df_tall : pandas.DataFrame
        Long-form DataFrame with columns ``dataset_name``, ``x``,
        ``mut_type``, ``mutation``, ``condition``, and ``mut_param``.
    x : str
        Column for the x-axis. Default ``'fusionreg'``.
    mut_param : str
        Column with mutation parameter values. Default ``'shift'``.
    width_scalar : int
        Width multiplier per facet column.
    height_scalar : int
        Height multiplier per facet row.

    Returns
    -------
    alt.FacetChart
    """
    highlight = alt.selection_point(on="mouseover", fields=["mutation"], nearest=True)
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


def shift_sparsity(
    sparsity_df,
    *,
    x="fusionreg",
    width_scalar=100,
    height_scalar=100,
):
    """Visualize shift parameter sparsity across regularization weights.

    Parameters
    ----------
    sparsity_df : pandas.DataFrame
        DataFrame with columns ``dataset_name``, ``x``, ``mut_type``,
        ``mut_param``, and ``sparsity``.
    x : str
        Column for the x-axis. Default ``'fusionreg'``.
    width_scalar : int
        Width multiplier per facet column.
    height_scalar : int
        Height multiplier per facet row.

    Returns
    -------
    alt.FacetChart
    """
    num_facet_rows = len(sparsity_df.dataset_name.unique())
    num_facet_cols = len(sparsity_df.mut_param.unique())

    base_chart = (
        alt.Chart(sparsity_df)
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
            y=alt.Y("sparsity", type="quantitative", title="Sparsity").axis(format="%"),
            color=alt.Color("mut_type", type="nominal", title="Mutation type"),
            tooltip=[
                x,
                "sparsity",
                "mut_type",
            ],
        )
        .properties(
            width=num_facet_cols * width_scalar,
            height=num_facet_rows * height_scalar,
        )
    )

    if sparsity_df[x].dtype.kind in "biufc":
        chart = base_chart.mark_point() + base_chart.mark_line()
    else:
        chart = base_chart.mark_bar().encode(xOffset="mut_type")

    return chart.facet(
        row=alt.Row("dataset_name", title="Dataset"),
        column=alt.Column("mut_param", title="Experimental Shifts"),
    )


def mut_param_dataset_correlation(
    replicate_df,
    *,
    x="fusionreg",
    r=1,
    width_scalar=400,
    height=400,
):
    """Visualize mutation parameter correlation between replicate datasets.

    Parameters
    ----------
    replicate_df : pandas.DataFrame
        DataFrame with columns ``datasets``, ``mut_param``,
        ``correlation``, and ``x``.
    x : str
        Column for the x-axis. Default ``'fusionreg'``.
    r : int
        Exponent used (1 for pearson, 2 for R^2). Only affects title.
    width_scalar : int
        Width multiplier per facet column.
    height : int
        Chart height.

    Returns
    -------
    alt.FacetChart
    """
    comparisons = replicate_df["datasets"].unique()
    title_suffix = "(R^2)" if r == 2 else "(pearson)"

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

    if replicate_df[x].dtype.kind in "biufc":
        chart = base_chart.mark_point() + base_chart.mark_line()
    else:
        chart = base_chart.mark_bar().encode(xOffset="mut_param")

    return chart.facet(
        column=alt.Column("datasets", title="Experiment comparison"),
    )


def replicate_param_scatter(
    muts_df_pair,
    *,
    x_col,
    y_col,
    x_label=None,
    y_label=None,
    color_by=None,
    point_size=30,
    point_opacity=0.5,
    width=500,
    height=500,
):
    """Scatter plot comparing mutation parameters between two replicates.

    Parameters
    ----------
    muts_df_pair : pandas.DataFrame
        One row per mutation with columns ``x_col``, ``y_col``.
        The ``mutation`` column (or index) contains mutation strings
        like "A5G". Optional columns for tooltips: ``site``, ``wildtype``,
        ``mutant`` (derived by parsing mutation strings via ``split_sub``).
    x_col, y_col : str
        Column names for the two datasets' parameter values.
    x_label, y_label : str or None
        Axis labels (default to column names).
    color_by : str or None
        Column to color points by.
    point_size, point_opacity : float
        Point styling.
    width, height : int
        Chart dimensions.

    Returns
    -------
    alt.LayerChart
        Scatter + diagonal identity line + Pearson r annotation.
    """
    df = muts_df_pair.copy()
    if x_label is None:
        x_label = x_col
    if y_label is None:
        y_label = y_col

    # Tooltips
    tooltip_cols = [x_col, y_col]
    if "mutation" in df.columns:
        tooltip_cols = ["mutation"] + tooltip_cols
    for col in ("site", "wildtype", "mutant"):
        if col in df.columns:
            tooltip_cols.append(col)
    if color_by and color_by not in tooltip_cols:
        tooltip_cols.append(color_by)

    # Scatter
    color_enc = alt.Color(f"{color_by}:N") if color_by else alt.value("#4682b4")
    scatter = (
        alt.Chart(df)
        .mark_circle(size=point_size, opacity=point_opacity)
        .encode(
            x=alt.X(f"{x_col}:Q", title=x_label),
            y=alt.Y(f"{y_col}:Q", title=y_label),
            color=color_enc,
            tooltip=tooltip_cols,
        )
    )

    # Diagonal identity line
    valid = df[[x_col, y_col]].dropna()
    lo = float(min(valid[x_col].min(), valid[y_col].min()))
    hi = float(max(valid[x_col].max(), valid[y_col].max()))
    line_df = pd.DataFrame({x_col: [lo, hi], y_col: [lo, hi]})
    diag = (
        alt.Chart(line_df)
        .mark_line(strokeDash=[4, 4], color="grey", strokeWidth=1.5)
        .encode(x=f"{x_col}:Q", y=f"{y_col}:Q")
    )

    # Pearson r annotation
    r_val, _ = stats.pearsonr(valid[x_col], valid[y_col])
    ann_df = pd.DataFrame(
        {
            "text": [f"r = {r_val:.3f}"],
            "x": [lo + (hi - lo) * 0.05],
            "y": [hi - (hi - lo) * 0.05],
        }
    )
    annotation = (
        alt.Chart(ann_df)
        .mark_text(align="left", fontSize=14, fontWeight="bold")
        .encode(x="x:Q", y="y:Q", text="text:N")
    )

    return (scatter + diag + annotation).properties(width=width, height=height)


def times_seen_hist(mutations_df, *, conditions=None, width=400, height=300):
    """Interactive histogram of mutation occurrence counts.

    Parameters
    ----------
    mutations_df : pandas.DataFrame
        Mutations DataFrame with ``times_seen_{condition}`` columns.
    conditions : list of str or None
        Condition names. If None, auto-detected from column names
        matching ``times_seen_*``.
    width : int
        Chart width in pixels.
    height : int
        Chart height in pixels.

    Returns
    -------
    alt.Chart
    """
    if conditions is None:
        conditions = [
            c.replace("times_seen_", "")
            for c in mutations_df.columns
            if c.startswith("times_seen_")
        ]

    ts_cols = [f"times_seen_{c}" for c in conditions]
    long_df = mutations_df[ts_cols].melt(var_name="condition", value_name="times_seen")
    long_df["condition"] = long_df["condition"].str.replace(
        "^times_seen_", "", regex=True
    )

    return (
        alt.Chart(long_df)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("times_seen:Q", bin=True, title="Times seen"),
            y=alt.Y("count()", title="Number of mutations"),
            color=alt.Color("condition:N"),
            tooltip=["condition:N", "times_seen:Q"],
        )
        .properties(width=width, height=height)
    )


def func_score_boxplot(variants_df, *, width=400, height=300):
    """Interactive boxplot of functional scores by condition.

    Parameters
    ----------
    variants_df : pandas.DataFrame
        Variants DataFrame with ``condition`` and ``func_score`` columns.
    width : int
        Chart width in pixels.
    height : int
        Chart height in pixels.

    Returns
    -------
    alt.Chart
    """
    return (
        alt.Chart(variants_df)
        .mark_boxplot()
        .encode(
            x=alt.X("condition:N", title="Condition"),
            y=alt.Y("func_score:Q", title="Functional score"),
            color=alt.Color("condition:N", legend=None),
            tooltip=["condition:N"],
        )
        .properties(width=width, height=height)
    )


def ge_landscape(
    variants_df,
    ge_curve_df,
    fitness_col=None,
    color_by="condition",
    point_size=5,
    point_opacity=0.3,
    curve_color="grey",
    curve_width=3,
    width=500,
    height=400,
    space="fitness",
):
    """Plot the global epistasis landscape.

    Overlays per-variant fitness scores on the global epistasis curve
    ``g(φ)``, showing how the nonlinear transformation maps latent
    phenotype to observed fitness. Wildtype latent phenotypes for each
    condition are shown as dashed vertical reference lines.

    Parameters
    ----------
    variants_df : pandas.DataFrame
        Variant-level data with columns ``predicted_latent``,
        ``condition``, ``wildtype_latent``, and ``fitness_col``.
        Typically from :meth:`Model.get_ge_landscape_df`.
    ge_curve_df : pandas.DataFrame
        Curve DataFrame. For ``space="fitness"`` it has columns
        ``predicted_latent`` and ``ge_curve_value``; for
        ``space="func_score"`` it is long-form with ``condition``,
        ``predicted_latent`` and ``func_score_curve_value``.
        Typically from :meth:`Model.get_ge_landscape_df`.
    fitness_col : str, optional
        Which variant column to scatter on the y-axis. When ``None`` (the
        default), resolves to the *measured* observation for each space:
        ``'measured_fitness'`` for ``space="fitness"`` and ``'func_score'``
        for ``space="func_score"``. The model's prediction is the curve
        itself (``predicted_func_score`` equals the per-condition curve
        evaluated at each variant's latent), so the scatter shows measured
        data and the curve shows the fit; the vertical gap is the residual.
        Pass ``'predicted_func_score'`` explicitly to overlay predictions.
    color_by : str
        Column to color scatter points by. Default is ``'condition'``.
    point_size : float
        Size of scatter points. Default is 5.
    point_opacity : float
        Opacity of scatter points. Default is 0.3.
    curve_color : str
        Color of the ``g(φ)`` curve. Default is ``'grey'``.
    curve_width : float
        Stroke width of the ``g(φ)`` curve. Default is 3.
    width : int
        Chart width in pixels. Default is 500.
    height : int
        Chart height in pixels. Default is 400.
    space : str
        ``"fitness"`` (default) draws the single shared ``g(φ)`` curve on
        ``ge_curve_value``. ``"func_score"`` draws one condition-colored
        curve per condition on ``func_score_curve_value`` and labels the
        y-axis "Functional score".

    Returns
    -------
    alt.LayerChart
        Altair layered chart with scatter, curve, and wildtype reference
        lines.
    """
    if space not in ("fitness", "func_score"):
        raise ValueError(f"space must be 'fitness' or 'func_score', got {space!r}")
    if fitness_col is None:
        # Default to the *measured* observation in each space, so the scatter
        # shows data (not model predictions) against the fitted curve:
        #   fitness    -> measured_fitness  (observed, in g(φ) space)
        #   func_score -> func_score        (observed functional score)
        fitness_col = "func_score" if space == "func_score" else "measured_fitness"
    y_title = "Functional score" if space == "func_score" else "Fitness"

    # Interactive legend selection to toggle conditions
    selection = alt.selection_point(fields=[color_by], bind="legend")

    # Scatter layer: variant fitness vs latent phenotype
    scatter = (
        alt.Chart(variants_df)
        .mark_circle(size=point_size, opacity=point_opacity)
        .encode(
            x=alt.X(
                "predicted_latent:Q",
                title="Predicted latent phenotype (φ)",
            ),
            y=alt.Y(f"{fitness_col}:Q", title=y_title),
            color=alt.Color(f"{color_by}:N"),
            opacity=alt.condition(
                selection,
                alt.value(point_opacity),
                alt.value(0),
            ),
        )
        .add_params(selection)
    )

    # Curve layer
    if space == "func_score":
        # one line per condition from α·(g(φ) − g(φ_wt))
        curve = (
            alt.Chart(ge_curve_df)
            .mark_line(strokeWidth=curve_width)
            .encode(
                x="predicted_latent:Q",
                y=alt.Y("func_score_curve_value:Q", title=y_title),
                color=alt.Color(f"{color_by}:N"),
                detail=alt.Detail(f"{color_by}:N"),
            )
        )
    else:
        curve = (
            alt.Chart(ge_curve_df)
            .mark_line(color=curve_color, strokeWidth=curve_width)
            .encode(
                x="predicted_latent:Q",
                y=alt.Y("ge_curve_value:Q"),
            )
        )

    # WT reference lines
    wt_data = (
        variants_df[["condition", "wildtype_latent"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    wt_rules = (
        alt.Chart(wt_data)
        .mark_rule(strokeDash=[4, 4])
        .encode(
            x="wildtype_latent:Q",
            color=alt.Color("condition:N"),
            opacity=alt.condition(
                selection,
                alt.value(0.6),
                alt.value(0),
            ),
        )
    )

    return (scatter + curve + wt_rules).properties(width=width, height=height)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
