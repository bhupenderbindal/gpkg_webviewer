"""Cluster Validation \u2014 Streamlit front-end.

Run from this folder::

    streamlit run streamlit_app.py

Same features as the notebook app (``cluster_validation_app.py``); plotting
and data loading come from ``cluster_core``.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI backend: figures are rendered to PNG

import importlib  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import streamlit as st  # noqa: E402

import cluster_core  # noqa: E402

# Streamlit re-runs this script on edits but can keep a stale cluster_core in
# memory; reloading it each run (a few ms) keeps the two files in sync.
importlib.reload(cluster_core)

from cluster_core import (  # noqa: E402
    DEFAULTS,
    EXCLUDE_COLS,
    OUT_DIR,
    SCALE_OPTIONS,
    SUPPORTED_EXT,
    _sorted_labels,
    clean_string,
    cluster_validation,
    gpd,
    list_layers_any,
    read_geodata,
)


# Last data path, kept on disk because a browser refresh clears session state.
LAST_PATH_FILE = Path(__file__).with_name(".last_data_path")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _read_last_path():
    try:
        return LAST_PATH_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _remember_path(p):
    try:
        if p != _read_last_path():
            LAST_PATH_FILE.write_text(p, encoding="utf-8")
    except OSError:
        pass  # not being able to remember the path must not break the app


@st.cache_data(show_spinner=False)
def _layers(source):
    return list_layers_any(source)


@st.cache_data(show_spinner="Loading data\u2026")
def _load(source, layer):
    return read_geodata(source, layer=layer)


def _preview_frame(gdf, n=5):
    """First *n* rows with geometry as WKT text, which st.dataframe can show."""
    head = gdf.head(n)
    if gpd is not None and isinstance(head, gpd.GeoDataFrame):
        return head.to_wkt()
    return pd.DataFrame(head)


def _fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _safe_name(i, title):
    return f"plot_{i}_{re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')}.png"


def _defaults(key, options):
    return [c for c in DEFAULTS.get(key, []) if c in options]


# A numeric hue with more distinct values than this gets a colour gradient
# instead of one colour per value.
MAX_HUE_CATEGORIES = 20
# Facet columns with more values than this would give an unreadable grid.
MAX_FACETS = 30


def _category_palette(labels):
    name = ("tab10" if len(labels) <= 10 else "tab20" if len(labels) <= 20
            else "husl")
    return dict(zip(labels, sns.color_palette(name, len(labels))))


def scatter_plot(gdf, x, y, hue=None, facet=None, facet_cols=3, share_axes=True,
                 xscale="linear", yscale="linear", point_size=20, alpha=0.7,
                 color="#1f77b4"):
    """Scatter of *y* against *x*, coloured by *hue* (categories or gradient)
    and split into one panel per value of *facet*.

    Colours, gradient range and axes limits (with *share_axes*) are the same
    in every panel, so the panels can be compared directly.
    """
    data = pd.DataFrame(gdf[[c for c in dict.fromkeys([x, y, hue, facet]) if c]])

    if facet is None:
        panels = [(None, data)]
    else:
        # Same sorted order as the other plots; rows without a facet value
        # belong to no panel.
        panels = [(v, data[data[facet] == v]) for v in _sorted_labels(data[facet])]
    ncols = max(1, min(facet_cols, len(panels)))
    nrows = -(-len(panels) // ncols)
    size = (8, 6) if facet is None else (4.5 * ncols, 3.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=size, squeeze=False,
                             sharex=share_axes, sharey=share_axes,
                             layout="constrained")
    axes = axes.ravel()

    gradient = (
        hue is not None
        and pd.api.types.is_numeric_dtype(data[hue])
        and data[hue].nunique() > MAX_HUE_CATEGORIES
    )
    if hue is not None and not gradient:
        # Categories keyed by their text so numbers and strings sort and
        # colour the same way as in the other plots.
        palette = _category_palette([str(v) for v in _sorted_labels(data[hue])])
    kws = dict(s=point_size, alpha=alpha, edgecolors="none")
    mappable = None

    for ax, (value, part) in zip(axes, panels):
        if hue is None:
            ax.scatter(part[x], part[y], color=color, **kws)
        elif gradient:
            mappable = ax.scatter(part[x], part[y], c=part[hue], cmap="viridis",
                                  vmin=data[hue].min(), vmax=data[hue].max(),
                                  **kws)
        else:
            keys = part[hue].map(lambda v: None if pd.isna(v) else str(v))
            for label, c in palette.items():
                sel = keys == label
                if sel.any():
                    ax.scatter(part.loc[sel, x], part.loc[sel, y], color=c, **kws)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.grid(alpha=0.3)
        if value is not None:
            ax.set_title(f"{value}  (n={len(part)})", fontsize=10)

    for ax in axes[len(panels):]:
        ax.set_visible(False)
    # Axis labels only on the outer panels, like a seaborn FacetGrid.
    for i, ax in enumerate(axes[:len(panels)]):
        if i % ncols == 0:
            ax.set_ylabel(clean_string(y))
        if i + ncols >= len(panels):
            ax.set_xlabel(clean_string(x))
            ax.tick_params(axis="x", labelbottom=True)

    if gradient:
        fig.colorbar(mappable, ax=axes[:len(panels)].tolist(),
                     label=clean_string(hue))
    elif hue is not None:
        handles = [
            plt.Line2D([], [], marker="o", linestyle="", color=c, label=label)
            for label, c in palette.items()
        ]
        fig.legend(handles=handles, title=clean_string(hue), loc="center left",
                   bbox_to_anchor=(1.0, 0.5), frameon=False)

    title = f"{clean_string(y)} vs {clean_string(x)}"
    if facet is not None:
        title += f" by {clean_string(facet)}"
    fig.suptitle(title)
    return fig


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Cluster Validation", layout="wide")
st.title("Cluster Validation")

# -- 1: load ------------------------------------------------------------------
st.header("1 \u2014 Load data")
upload = st.file_uploader(
    "Choose file", type=[e.lstrip(".") for e in SUPPORTED_EXT]
)
path = st.text_input(
    "\u2026or data path",
    value=_read_last_path(),
    placeholder=r"C:\data\clusters.gpkg  (.csv .gpkg .geojson .shp .zip ...)",
    help="The last path that loaded successfully is restored after a refresh.",
)

if upload is not None:
    source = (upload.name, upload.getvalue())
elif path.strip():
    source = path.strip()
else:
    _remember_path("")  # box cleared on purpose: forget the old path
    st.info(f"No file loaded \u2014 accepts {', '.join(SUPPORTED_EXT)}")
    st.stop()

layers = _layers(source)
layer = st.selectbox("Layer", layers) if len(layers) > 1 else None

try:
    gdf = _load(source, layer)
except Exception as e:  # noqa: BLE001
    st.error(f"\u2717 {e}")
    st.stop()

if isinstance(source, str):
    _remember_path(source)

all_cols = [c for c in gdf.columns if c not in {"geometry", "field_1"}]
numeric_cols = [
    c for c in gdf.select_dtypes(include="number").columns if c not in EXCLUDE_COLS
]
# Percentage plots and hue count categories, so string columns are valid there.
category_cols = [c for c in all_cols if c not in EXCLUDE_COLS]

geo_note = ""
if gpd is not None and isinstance(gdf, gpd.GeoDataFrame):
    try:
        kinds = ", ".join(sorted(gdf.geom_type.dropna().unique()))
    except Exception:  # noqa: BLE001
        kinds = "geometry"
    geo_note = f" \u2022 {kinds} ({gdf.crs.to_string() if gdf.crs else 'no CRS'})"
st.success(f"\u2713 Loaded {len(gdf)} rows, {len(all_cols)} columns{geo_note}")

with st.expander("Columns, dtypes and first 5 rows", expanded=True):
    left, right = st.columns([1, 3])
    with left:
        st.markdown("**Columns & dtypes**")
        st.dataframe(
            pd.DataFrame(
                {"column": gdf.columns, "dtype": gdf.dtypes.astype(str).values}
            ),
            hide_index=True,
        )
    with right:
        st.markdown("**First 5 rows**")
        st.dataframe(_preview_frame(gdf))

# -- 2: config ----------------------------------------------------------------
st.header("2 \u2014 Configure parameters")
cluster_default = _defaults("cluster_col", all_cols)
cluster_col = st.selectbox(
    "Cluster column",
    all_cols,
    index=all_cols.index(cluster_default[0]) if cluster_default else 0,
)

c1, c2, c3 = st.columns(3)
with c1:
    boxplot_cols = st.multiselect(
        "Boxplot columns", numeric_cols, default=_defaults("boxplot", numeric_cols)
    )
with c2:
    pct_cols = st.multiselect(
        "Percentage plot columns", category_cols, default=_defaults("pct", category_cols)
    )
with c3:
    sum_cols = st.multiselect(
        "Sum plot columns", numeric_cols, default=_defaults("sum", numeric_cols)
    )

scale_map = {}
if boxplot_cols:
    with st.expander("Boxplot Y-axis scale"):
        scale_cols = st.columns(3)
        for i, col in enumerate(boxplot_cols):
            with scale_cols[i % 3]:
                scale_map[col] = st.selectbox(col, SCALE_OPTIONS, key=f"scale_{col}")

c1, c2, c3 = st.columns(3)
with c1:
    hue_options = [None] + category_cols
    hue_default = _defaults("hue", category_cols)
    boxplot_hue = st.selectbox(
        "Boxplot hue",
        hue_options,
        index=hue_options.index(hue_default[0]) if hue_default else 0,
        format_func=lambda v: "(none)" if v is None else v,
    )
    boxplot_color = st.color_picker(
        "Boxplot colour", "#7c7c0a", disabled=boxplot_hue is not None,
        help="Ignored when a hue column is chosen.",
    )
with c2:
    sum_color = st.color_picker("Sum plot colour", "#e7969c")
    xlabel_rotation = st.slider(
        "X-axis label rotation (°)", -90, 90, 0, step=5,
        help="Cluster labels on the percentage plots and boxplots.",
    )
with c3:
    use_random = st.checkbox("Use random clusters")
    save_local = st.checkbox(f"Also save PNGs to {OUT_DIR}/")

st.subheader("Scatter plot")
show_scatter = st.checkbox("Add a scatter plot", disabled=len(numeric_cols) < 1)
if show_scatter:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        y_default = _defaults("boxplot", numeric_cols)
        scatter_x = st.selectbox("X column", numeric_cols, key="scatter_x")
        scatter_y = st.selectbox(
            "Y column", numeric_cols, key="scatter_y",
            index=numeric_cols.index(y_default[0]) if y_default else 0,
        )
    with c2:
        scatter_hue = st.selectbox(
            "Scatter hue",
            [None] + all_cols,
            index=1 + all_cols.index(cluster_col),
            format_func=lambda v: "(none)" if v is None else v,
            key="scatter_hue",
            help="Defaults to the cluster column. Numeric columns with more "
            f"than {MAX_HUE_CATEGORIES} values are shown as a colour gradient.",
        )
        scatter_color = st.color_picker(
            "Point colour", "#1f77b4", disabled=scatter_hue is not None,
            help="Ignored when a hue column is chosen.",
        )
    with c3:
        scatter_xscale = st.selectbox("X scale", SCALE_OPTIONS, key="scatter_xscale")
        scatter_yscale = st.selectbox("Y scale", SCALE_OPTIONS, key="scatter_yscale")
        scatter_size = st.slider("Point size", 2, 100, 20, key="scatter_size")
        scatter_alpha = st.slider("Opacity", 0.1, 1.0, 0.7, step=0.05,
                                  key="scatter_alpha")
    with c4:
        scatter_facet = st.selectbox(
            "Facet by",
            [None] + all_cols,
            format_func=lambda v: "(none)" if v is None else v,
            key="scatter_facet",
            help="One panel per value of this column, laid out in a grid. "
            f"At most {MAX_FACETS} values.",
        )
        facet_cols = st.slider("Panels per row", 1, 6, 3, key="scatter_facet_cols",
                               disabled=scatter_facet is None)
        share_axes = st.checkbox("Same axis ranges in every panel", value=True,
                                 key="scatter_share", disabled=scatter_facet is None)

# -- 3: run -------------------------------------------------------------------
st.header("3 \u2014 Run")
if st.button("Run Cluster Validation", type="primary"):
    for label, cols in (
        ("boxplot", boxplot_cols),
        ("percentage plot", pct_cols),
        ("sum plot", sum_cols),
    ):
        if not cols:
            st.warning(f"No {label} columns selected \u2014 skipping.")
    hue_checks = pct_cols + ([boxplot_hue] if boxplot_hue else [])
    if show_scatter and scatter_hue and not (
        pd.api.types.is_numeric_dtype(gdf[scatter_hue])
        and gdf[scatter_hue].nunique() > MAX_HUE_CATEGORIES
    ):
        hue_checks.append(scatter_hue)
    for c in dict.fromkeys(hue_checks):
        n = gdf[c].nunique()
        if n > 20:
            st.warning(
                f"'{c}' has {n} distinct values \u2014 the plot will be crowded; "
                "pick a categorical column."
            )

    try:
        with st.spinner("Running cluster_validation\u2026"):
            figures = cluster_validation(
                gdf=gdf,
                boxplot_columns=boxplot_cols,
                percentage_plot_columns=pct_cols,
                sum_plot_columns=sum_cols,
                bin_to_label=None,
                cluster_col=cluster_col,
                if_random=use_random,
                boxplot_yscale=scale_map,
                boxplot_color=boxplot_color,
                sumplot_color=sum_color,
                boxplot_hue=boxplot_hue,
                xlabel_rotation=xlabel_rotation,
            )
        n_facets = gdf[scatter_facet].nunique() if show_scatter and scatter_facet else 0
        if n_facets > MAX_FACETS:
            st.error(f"Scatter skipped: '{scatter_facet}' has {n_facets} distinct "
                     f"values (max {MAX_FACETS} panels); pick a categorical column.")
        elif show_scatter:
            for axis, col, scale in (
                ("X", scatter_x, scatter_xscale), ("Y", scatter_y, scatter_yscale)
            ):
                n = int((gdf[col] <= 0).sum())
                if scale == "log" and n:
                    st.warning(f"{axis} log scale: {n} value(s) of '{col}' "
                               "are ≤ 0 and are not shown.")
            fig = scatter_plot(
                gdf, scatter_x, scatter_y, hue=scatter_hue, facet=scatter_facet,
                facet_cols=facet_cols, share_axes=share_axes,
                xscale=scatter_xscale, yscale=scatter_yscale,
                point_size=scatter_size, alpha=scatter_alpha, color=scatter_color,
            )
            title = f"Scatter — {scatter_y} vs {scatter_x}"
            if scatter_facet:
                title += f" by {scatter_facet}"
            figures.append((title, fig))
    except Exception as e:  # noqa: BLE001
        st.error(f"ERROR: {e}")
        st.session_state.pop("results", None)
    else:
        results = [
            (title, _safe_name(i, title), _fig_to_png(fig))
            for i, (title, fig) in enumerate(figures, 1)
        ]
        if save_local and results:
            OUT_DIR.mkdir(exist_ok=True)
            for _, name, png in results:
                (OUT_DIR / name).write_bytes(png)
            st.info(f"Saved {len(results)} PNG(s) to {OUT_DIR}/")
        st.session_state["results"] = results

# Results are kept in session state so they survive widget-triggered reruns.
results = st.session_state.get("results")
if results:
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w") as zf:
        for _, name, png in results:
            zf.writestr(name, png)
    st.download_button(
        "Download all plots (.zip)",
        zbuf.getvalue(),
        file_name="cluster_validation_plots.zip",
        mime="application/zip",
    )
    for title, name, png in results:
        st.subheader(title)
        st.image(png)
        st.download_button(
            "Download PNG", png, file_name=name, mime="image/png", key=f"dl_{name}"
        )
