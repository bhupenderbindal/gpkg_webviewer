"""Cluster validation core: data loading and plotting, with no UI code.

Shared by the Jupyter front-end (``cluster_validation_app.py``) and the
Streamlit front-end (``streamlit_app.py``). Loads geospatial formats
(GeoPackage, GeoJSON, Shapefile, FlatGeobuf, GeoParquet, ...) in addition to
plain CSV. Geometry is ignored by the plots; only the attribute columns are
used.

Dependencies assumed available: numpy, pandas, matplotlib, seaborn,
geopandas (for .gpkg / .geojson / .shp / ...).
"""

from __future__ import annotations

import io
import math
import re
import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover - geopandas is expected in the env
    gpd = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXCLUDE_COLS = {"field_1", "TOPOI_ID", "geometry"}
OUT_DIR = Path("cluster_validation_plots")
SCALE_OPTIONS = ["linear", "log", "symlog"]

# Formats read straight through pandas; everything else goes via geopandas.
_TABULAR_EXT = {".csv"}
_GEO_EXT = {
    ".gpkg",
    ".geojson",
    ".json",
    ".shp",
    ".zip",
    ".gml",
    ".kml",
    ".fgb",
    ".parquet",
    ".gpq",
}
MULTI_LAYER_EXT = {".gpkg", ".gml", ".kml"}
SUPPORTED_EXT = sorted(_TABULAR_EXT | _GEO_EXT)
UPLOAD_ACCEPT = ",".join(SUPPORTED_EXT)
DEFAULTS = {
    "cluster_col": ["type"],
    "boxplot": ["LST_avg"],
    "hue": ["city"],
    "pct": ["richness"],
    "sum": ["built_up_area"],
}


# ---------------------------------------------------------------------------
# Core plotting  (port of cluster_validation_browser)
# ---------------------------------------------------------------------------
def clean_string(s: str) -> str:
    s = re.sub(r"[_-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s.capitalize()


def _sorted_labels(values):
    """Sorted unique non-null labels; works for numbers, strings, or a mix."""
    labels = pd.Series(values).dropna().unique().tolist()
    try:
        return sorted(labels)
    except TypeError:  # mixed types, e.g. 1 and "a"
        return sorted(labels, key=str)


def _rotate_xticks(ax, angle):
    """Rotate x tick labels; anchor them at their right end so they stay
    under their tick when tilted."""
    if not angle:
        return
    ax.tick_params(axis="x", labelrotation=angle)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right" if 0 < angle < 90 else
                                    "left" if -90 < angle < 0 else "center")
        # lbl.set_rotation_mode("anchor")


def cluster_validation(
    gdf,
    boxplot_columns,
    percentage_plot_columns,
    sum_plot_columns,
    bin_to_label,
    cluster_col,
    if_random=False,
    boxplot_yscale=None,
    boxplot_color="#7c7c0a",
    sumplot_color="#e7969c",
    boxplot_hue=None,
    xlabel_rotation=0,
):
    """Return a list of ``(title, matplotlib.figure.Figure)`` tuples.

    *xlabel_rotation* (degrees) rotates the cluster labels on the x-axis of
    the percentage plots and boxplots (sum plots have clusters on the y-axis).
    """
    if boxplot_yscale is None:
        boxplot_yscale = {}

    figures = []

    if if_random:
        num_cl = gdf[cluster_col].nunique()
        cluster_col = "cluster_num_rand"
        gdf = gdf.copy()
        gdf[cluster_col] = np.random.randint(0, num_cl, gdf.shape[0])

    # One explicit order for every plot: seaborn otherwise orders string
    # labels by first appearance, which misaligns the count/mean annotations.
    order = _sorted_labels(gdf[cluster_col])

    plt.style.use("default")

    # -- percentage stacked-bar plots ------------------------------------
    for col_x in percentage_plot_columns:
        count_df = pd.crosstab(gdf[cluster_col], gdf[col_x])
        count_df = count_df.reindex(
            [c for c in order if c in count_df.index]
        )[_sorted_labels(count_df.columns)]
        pct_tbl = count_df.div(count_df.sum(axis=1), axis=0)
        category_count_dict = {}
        for r in pct_tbl.transpose().iterrows():
            category_count_dict[r[0]] = list(r[1])
        species = tuple(str(c) for c in count_df.index)
        width = 0.6
        category_ = list(gdf[col_x])
        bin_to_label_use = bin_to_label or dict(zip(category_, category_))

        with mpl.rc_context({"font.size": 4, "font.weight": 2}):
            fig, ax = plt.subplots(figsize=(7, 4))
            bottom = np.zeros(len(species))
            bin_colors = sns.color_palette("tab20", len(category_count_dict))
            for ci, (category_, category_count) in enumerate(
                category_count_dict.items()
            ):
                p = ax.bar(
                    species,
                    category_count,
                    width,
                    label=bin_to_label_use.get(category_, str(category_)),
                    bottom=bottom,
                    color=bin_colors[ci],
                )
                bottom += np.array(category_count)
                ax.bar_label(p, label_type="center")
            ax.set_title(f"{clean_string(col_x)} distribution in clusters")
            _rotate_xticks(ax, xlabel_rotation)
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
            fig.tight_layout(rect=[0, 0, 0.80, 1])
            figures.append((f"Percentage — {col_x}", fig))

    # -- boxplots ------------------------------------------------------------
    valid_bp = [c for c in boxplot_columns if c in gdf.columns]
    if valid_bp:
        cols = 2
        rows = math.ceil(len(valid_bp) / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(15, 10), layout="constrained")
        axs_flat = np.array(axs).flatten()
        if rows * cols - len(valid_bp) == 1:
            axs_flat[-1].axis("off")

        use_hue = bool(boxplot_hue) and boxplot_hue in gdf.columns
        if use_hue:
            hue_order = _sorted_labels(gdf[boxplot_hue])
            colour_kw = {
                "hue": boxplot_hue,
                "hue_order": hue_order,
                "palette": sns.color_palette(
                    "tab10" if len(hue_order) <= 10 else "tab20", len(hue_order)
                ),
            }
        else:
            colour_kw = {"color": boxplot_color}

        for ai, (ax, col) in enumerate(zip(axs_flat, valid_bp)):
            scale = boxplot_yscale.get(col, "linear")
            sns.boxplot(
                x=cluster_col,
                y=col,
                order=order,
                orient="x",
                width=0.8 if use_hue else 0.5,
                data=gdf,
                ax=ax,
                medianprops={"color": "r", "linewidth": 1},
                showmeans=True,
                meanprops={"markerfacecolor": "none"},
                **colour_kw,
            )
            # one shared legend (on the first panel) instead of one per panel
            if use_hue and ax.get_legend() is not None:
                if ai == 0:
                    sns.move_legend(
                        ax, "upper left", bbox_to_anchor=(1.01, 1),
                        title=clean_string(boxplot_hue), fontsize="x-small",
                    )
                else:
                    ax.get_legend().remove()
            lbl = clean_string(col)
            if scale != "linear":
                ax.set_yscale(scale)
                lbl += f" ({scale} scale)"
            ax.set_ylabel(lbl)
            _rotate_xticks(ax, xlabel_rotation)

            if not use_hue:
                ylo, yhi = ax.get_ylim()
                yrange = yhi - ylo if (yhi - ylo) != 0 else 1.0
                counts = gdf[cluster_col].value_counts()
                for i, cat in enumerate(order):
                    count = counts.get(cat, 0)
                    ax.text(i, yhi + yrange * 0.05, str(count), ha="center", size="x-small")
                ax.set_ylim(ylo, yhi + yrange * 0.25)

        figures.append(("Boxplots — cluster properties", fig))

    # -- sum bar plots -----------------------------------------------------
    valid_sp = [c for c in sum_plot_columns if c in gdf.columns]
    if valid_sp:
        cols = 2
        rows = math.ceil(len(valid_sp) / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(15, 10), layout="constrained")
        axs_flat = np.array(axs).flatten()
        if rows * cols - len(valid_sp) == 1:
            axs_flat[-1].axis("off")

        for ax, col in zip(axs_flat, valid_sp):
            sns.barplot(
                gdf,
                y=cluster_col,
                x=col,
                order=order,
                estimator="sum",
                errorbar=None,
                orient="y",
                color=sumplot_color,
                ax=ax,
            )
            gp = gdf.groupby(cluster_col).agg(
                **{f"sum_{col}": (col, "sum"), f"mean_{col}": (col, "mean")}
            ).reindex(order)
            ax.bar_label(
                ax.containers[0],
                labels=[f"{round(x, 2)} " for x in gp[f"mean_{col}"]],
                fontsize=5,
            )
            ax.set_xlabel(f"total {clean_string(col)}")

        figures.append(("Sum bar plots — cluster properties", fig))

    return figures


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _require_geopandas(ext):
    if gpd is None:
        raise RuntimeError(
            f"Reading '{ext}' needs geopandas. Install it (pip install geopandas) "
            "or load a .csv instead."
        )


# Encodings tried, in order, when a CSV is not valid UTF-8.
_CSV_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def _read_csv_any_encoding(filepath_or_buffer):
    """``pd.read_csv`` with a fallback chain for non-UTF-8 files."""
    seekable = hasattr(filepath_or_buffer, "seek")
    for enc in _CSV_ENCODINGS:
        try:
            if seekable:
                filepath_or_buffer.seek(0)
            return pd.read_csv(filepath_or_buffer, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    # last resort: replace undecodable bytes so the load still succeeds
    if seekable:
        filepath_or_buffer.seek(0)
    return pd.read_csv(
        filepath_or_buffer, encoding="latin-1", encoding_errors="replace"
    )


def list_layers(source):
    """Return the layer names in a multi-layer source (e.g. GeoPackage), else []."""
    ext = Path(str(source)).suffix.lower()
    if ext not in _GEO_EXT or gpd is None:
        return []
    try:
        import fiona  # geopandas' engine; ships with it

        return list(fiona.listlayers(source))
    except Exception:  # noqa: BLE001 - single-layer formats / older stacks
        try:
            return list(gpd.list_layers(source)["name"])
        except Exception:  # noqa: BLE001
            return []


def list_layers_any(source):
    """``list_layers`` for a path or a ``(name, bytes)`` upload tuple.

    Only multi-layer formats are probed; uploads are written to a temp file
    because the drivers need a path.
    """
    name = source[0] if isinstance(source, tuple) else str(source)
    ext = Path(name).suffix.lower()
    if ext not in MULTI_LAYER_EXT:
        return []
    if not isinstance(source, tuple):
        return list_layers(source)
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(source[1])
        probe = tmp.name
    try:
        return list_layers(probe)
    finally:
        try:
            Path(probe).unlink()
        except OSError:
            pass


def read_geodata(source, layer=None):
    """Load *source* (path or bytes-with-name tuple) into a (Geo)DataFrame.

    Supports .csv via pandas and .gpkg/.geojson/.shp/.zip/.parquet/... via
    geopandas. For byte uploads the content is written to a temp file first
    because most geo drivers cannot read from an in-memory buffer.
    """
    if isinstance(source, tuple):  # (name, data) from an upload widget
        name, data = source
        ext = Path(name).suffix.lower()
        if ext in _TABULAR_EXT:
            return _read_csv_any_encoding(io.BytesIO(data))
        _require_geopandas(ext)
        with tempfile.NamedTemporaryFile(suffix=ext or ".dat", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            return read_geodata(tmp_path, layer=layer)
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass

    path = str(source)
    ext = Path(path).suffix.lower()
    if ext in _TABULAR_EXT:
        return _read_csv_any_encoding(path)
    if ext in {".parquet", ".gpq"}:
        _require_geopandas(ext)
        return gpd.read_parquet(path)
    if ext in _GEO_EXT:
        _require_geopandas(ext)
        return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)

    # Unknown extension: try geopandas first (handles .gdb, .tab, no-ext dirs,
    # OGR connection strings), then fall back to a CSV read.
    if gpd is not None:
        try:
            return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
        except Exception:  # noqa: BLE001
            pass
    return _read_csv_any_encoding(path)
