"""Guards which dashboard cells re-run when a UI element's value changes.

Marimo re-runs, for a changed element bound to ``name``::

    get_referring_cells(name) - get_defining_cells(name)

(``marimo/_runtime/runtime.py``, in ``set_ui_element_value``). So the cost of
clicking a checkbox is decided entirely by how many *other* cells name the
element — a property of ``experiments/dashboard.py``'s dataflow graph, not of
anything the chart code does.

That makes the gating in ``dashboard.py`` statically checkable, which is the
only practical way to check it: reproducing the bug at runtime needs a live
kernel, a browser, and a fitted ``ModelCollection`` that no CI runner has.
``test_dashboard_discovery.py`` guards a different marimo invariant over the
same notebook in the same spirit.

The counts asserted here are exact rather than upper bounds so that both a
regression (a new cell starts naming a gated element) and a silent rename (the
element disappears and the check becomes vacuous) fail.
"""

import ast
from pathlib import Path

import pytest

#: The marimo notebook whose dataflow graph is under test.
_DASHBOARD_PY = Path(__file__).resolve().parents[1] / "experiments" / "dashboard.py"

#: Elements whose value must reach the kernel without running a single cell.
#: The four multi-select fit tables and the threshold slider are staged and
#: read at Plot-press time; the buttons act only through their ``on_change``
#: setters; ``summary_table``'s selection has no consumer at all.
_INERT = (
    "conv_table",
    "corr_table",
    "scatter_table",
    "sparsity_table",
    "times_seen_threshold_slider",
    "conv_plot_button",
    "corr_plot_button",
    "scatter_plot_button",
    "sparsity_plot_button",
    "summary_table",
)

#: Elements that are deliberately live, mapped to the variable defined by the
#: single cell allowed to refer to them. GE Landscape is single-select and
#: cheap, so it renders on selection; the scatter parameter dropdown must
#: re-render the chart without a second Plot press.
_LIVE = {
    "ge_table": "ge_chart",
    "scatter_param_dropdown": "scatter_chart",
}


def _cells_from_marimo():
    """Cells as ``(refs, defs)`` pairs from marimo's own dependency graph.

    This is the graph the runtime itself consults, so a check built on it
    cannot drift from the behaviour it guards.

    Returns:
        List of ``(refs, defs)`` tuples, both sets of names, one per cell.

    Raises:
        ImportError: If marimo's notebook loader is unavailable.
    """
    from marimo._ast.load import load_app

    app = load_app(str(_DASHBOARD_PY))
    return [
        (set(data.cell.refs), set(data.cell.defs))
        for data in app._cell_manager._cell_data.values()
        if data.cell is not None
    ]


def _cells_from_ast():
    """Cells as ``(refs, defs)`` pairs parsed from the notebook source.

    Fallback for the case where marimo's internal loader API moves. A cell
    function's *parameters* are its refs, which is the half this module's
    invariant is stated in. Defs come back empty: the ``return`` tuple is a
    serialization artifact that marimo strips before compiling a cell (see
    ``_ast/compiler.py``'s ``cell_factory``), so it is not a sound source of
    definitions and is deliberately not read here.

    Returns:
        List of ``(refs, defs)`` tuples, where every ``defs`` is empty.
    """
    tree = ast.parse(_DASHBOARD_PY.read_text())
    cells = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any(
            isinstance(d, ast.Attribute) and d.attr == "cell"
            for d in node.decorator_list
        ):
            continue
        cells.append(({a.arg for a in node.args.args}, set()))
    return cells


@pytest.fixture(scope="module")
def cells():
    """Marimo's real graph, falling back to the source parse."""
    try:
        return _cells_from_marimo()
    except ImportError:  # pragma: no cover - exercised only if the API moves
        return _cells_from_ast()


def _referrers(cells, name):
    """Cells marimo would re-run when the element bound to ``name`` changes."""
    return [refs for refs, defs in cells if name in refs and name not in defs]


def _definers(cells, name):
    """Cells that define ``name``."""
    return [defs for refs, defs in cells if name in defs]


@pytest.mark.parametrize("name", _INERT)
def test_inert_elements_have_no_referring_cells(cells, name):
    """Interacting with a staged element must run zero cells.

    A referrer here is the whole bug this guards: before #304 the cell
    building the Plot buttons named all four tables and the slider, so every
    checkbox click rebuilt the buttons and re-rendered the tab bar.
    """
    referrers = _referrers(cells, name)
    assert referrers == [], (
        f"{name!r} is named by {len(referrers)} other cell(s), so changing it "
        f"re-runs them. Move the reference into {name!r}'s own cell, or "
        f"consume a pre-composed container instead."
    )


@pytest.mark.parametrize("name", sorted(_INERT) + sorted(_LIVE))
def test_guarded_elements_still_exist(cells, name):
    """Every guarded name is defined exactly once.

    Without this, renaming an element away would make its referrer assertion
    vacuously true rather than failing.
    """
    if not any(defs for _, defs in cells):
        pytest.skip("defs unavailable on the AST fallback path")
    assert len(_definers(cells, name)) == 1


@pytest.mark.parametrize("name,rendered_by", sorted(_LIVE.items()))
def test_live_elements_have_exactly_one_referring_cell(cells, name, rendered_by):
    """A live element re-renders its own chart and nothing else.

    Exactly one referrer, and it must be the cell producing the chart — not
    the ``mo.ui.tabs`` layout cell, whose re-render is the churn #304 removes.
    """
    referrers = _referrers(cells, name)
    assert len(referrers) == 1, (
        f"{name!r} should be named by exactly the cell defining "
        f"{rendered_by!r}, but is named by {len(referrers)} cell(s)."
    )
    if not any(defs for _, defs in cells):
        return
    (consumer,) = [defs for refs, defs in cells if name in refs and name not in defs]
    assert rendered_by in consumer


def test_layout_cell_consumes_only_defined_names(cells):
    """Every name the tab-assembly cell reads is defined by some cell.

    An unbound name there is not a one-panel failure: every panel and chart
    feeds the single ``mo.ui.tabs`` call, so one ``NameError`` erases the
    entire tab bar. ``mo.stop`` in a chart cell would do the same, which is
    why the chart cells branch instead of halting.
    """
    if not any(defs for _, defs in cells):
        pytest.skip("defs unavailable on the AST fallback path")
    # Identify the cell structurally, by the panels it assembles, rather than
    # as "the def-less cell with the most refs" -- that heuristic would
    # silently retarget this assertion onto some future def-less cell.
    layout = [refs for refs, defs in cells if not defs and "conv_panel" in refs]
    assert len(layout) == 1, (
        "expected exactly one def-less cell referencing 'conv_panel' (the "
        f"mo.ui.tabs layout cell), found {len(layout)}."
    )
    defined = set().union(*(defs for _, defs in cells))
    missing = sorted(n for n in layout[0] if n not in defined and n != "mo")
    assert missing == [], f"layout cell reads undefined name(s): {missing}"
