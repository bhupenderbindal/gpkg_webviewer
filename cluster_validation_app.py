"""Cluster Validation GUI app.

Jupyter reproduction of the ``cluster-vis-py-index.html`` interface, extended
to load geospatial formats (GeoPackage, GeoJSON, Shapefile, FlatGeobuf,
GeoParquet, ...) in addition to plain CSV. Geometry is ignored by the plots;
only the attribute columns are used.

Usage (in a notebook cell)::

    from cluster_validation_app import launch
    launch()

All logic lives here so the notebook can stay a single hidden line and read
as a GUI app. Dependencies assumed available: numpy, pandas, matplotlib,
seaborn, ipywidgets, geopandas (for .gpkg / .geojson / .shp / ...).
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
import ipywidgets as widgets
from IPython.display import clear_output, display

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover - geopandas is expected in the Jupyter env
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
SUPPORTED_EXT = sorted(_TABULAR_EXT | _GEO_EXT)
UPLOAD_ACCEPT = ",".join(SUPPORTED_EXT)
DEFAULTS = {
    "cluster_col": ["depth_300"],
    "boxplot": [
        "total_buildings",
        "total_vol",
        "settlement_perimeter",
        "compactness",
        "built_up_area",
        "agri_ratio",
    ],
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
):
    """Return a list of ``(title, matplotlib.figure.Figure)`` tuples."""
    if boxplot_yscale is None:
        boxplot_yscale = {}

    figures = []

    if if_random:
        num_cl = int(np.max(gdf["cluster_num"]))
        cluster_col = "cluster_num_rand"
        gdf = gdf.copy()
        gdf[cluster_col] = np.random.randint(0, num_cl, gdf.shape[0])

    plt.style.use("default")

    # -- percentage stacked-bar plots ------------------------------------
    for col_x in percentage_plot_columns:
        count_df = pd.crosstab(gdf[cluster_col], gdf[col_x])
        pct_tbl = count_df.div(count_df.sum(axis=1), axis=0)
        category_count_dict = {}
        for r in pct_tbl.transpose().iterrows():
            category_count_dict[r[0]] = list(r[1])
        species = tuple(count_df.index)
        width = 0.6
        category_ = list(gdf[col_x])
        bin_to_label_use = bin_to_label or dict(zip(category_, category_))

        with mpl.rc_context({"font.size": 4, "font.weight": 2}):
            fig, ax = plt.subplots(figsize=(7, 4))
            bottom = np.zeros(len(species))
            bin_colors = sns.color_palette("tab20", len(category_count_dict))
            for category_, category_count in category_count_dict.items():
                p = ax.bar(
                    species,
                    category_count,
                    width,
                    label=bin_to_label_use.get(category_, str(category_)),
                    bottom=bottom,
                    color=bin_colors[category_ - 1],
                )
                bottom += np.array(category_count)
                ax.bar_label(p, label_type="center")
            ax.set_title(f"{clean_string(col_x)} distribution in clusters")
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
            fig.tight_layout(rect=[0, 0, 0.80, 1])
            figures.append((f"Percentage \u2014 {col_x}", fig))

    # -- boxplots ------------------------------------------------------------
    valid_bp = [c for c in boxplot_columns if c in gdf.columns]
    if valid_bp:
        cols = 2
        rows = math.ceil(len(valid_bp) / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(15, 10), layout="constrained")
        axs_flat = np.array(axs).flatten()
        if rows * cols - len(valid_bp) == 1:
            axs_flat[-1].axis("off")

        for ax, col in zip(axs_flat, valid_bp):
            scale = boxplot_yscale.get(col, "linear")
            sns.boxplot(
                x=cluster_col,
                y=col,
                orient="x",
                width=0.5,
                data=gdf,
                ax=ax,
                medianprops={"color": "r", "linewidth": 1},
                showmeans=True,
                meanprops={"markerfacecolor": "none"},
                color=boxplot_color,
            )
            lbl = clean_string(col)
            if scale != "linear":
                ax.set_yscale(scale)
                lbl += f" ({scale} scale)"
            ax.set_ylabel(lbl)
            ylo, yhi = ax.get_ylim()
            yrange = yhi - ylo if (yhi - ylo) != 0 else 1.0
            for i, cat in enumerate(np.sort(gdf[cluster_col].unique())):
                count = len(gdf[gdf[cluster_col] == cat])
                ax.text(i, yhi + yrange * 0.05, str(count), ha="center", size="x-small")
            ax.set_ylim(ylo, yhi + yrange * 0.25)

        figures.append(("Boxplots \u2014 cluster properties", fig))

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
                estimator="sum",
                errorbar=None,
                orient="y",
                color=sumplot_color,
                ax=ax,
            )
            gp = gdf.groupby(cluster_col).agg(
                **{f"sum_{col}": (col, "sum"), f"mean_{col}": (col, "mean")}
            )
            ax.bar_label(
                ax.containers[0],
                labels=[f"{round(x, 2)} " for x in gp[f"mean_{col}"]],
                fontsize=5,
            )
            ax.set_xlabel(f"total {clean_string(col)}")

        figures.append(("Sum bar plots \u2014 cluster properties", fig))

    return figures


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
def _section(label, *children):
    return widgets.VBox(
        [widgets.HTML(f"<b style='color:#1e3a5f'>{label}</b>"), *children],
        layout=widgets.Layout(
            border="1px solid #e0e7f0", padding="10px 14px", margin="0 0 12px 0"
        ),
    )


def _upload_item(file_upload):
    """Return ``(name, bytes)`` for the single uploaded file, or ``None``."""
    val = file_upload.value
    if not val:
        return None
    item = list(val.values())[0] if isinstance(val, dict) else val[0]
    return item["name"], bytes(item["content"])


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


def build_cluster_validation_app():
    """Build and return the widget tree (does not display it)."""
    state = {"gdf": None}

    title = widgets.HTML(
        "<h2 style='color:#1e3a5f;border-bottom:2px solid #89b4fa;"
        "padding-bottom:.3rem;margin:0 0 .6rem 0'>Cluster Validation</h2>"
    )

    # -- 1: load ----------------------------------------------------------
    path_box = widgets.Text(
        placeholder=r"optional: C:\data\clusters.gpkg  (.csv .gpkg .geojson .shp .zip ...)",
        description="Data path",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="520px"),
    )
    file_upload = widgets.FileUpload(
        accept=UPLOAD_ACCEPT, multiple=False, description="Choose file"
    )
    layer_dd = widgets.Dropdown(
        description="Layer",
        options=[],
        style={"description_width": "initial"},
        layout=widgets.Layout(width="360px", display="none"),
    )
    load_btn = widgets.Button(description="Load data", button_style="primary")
    load_status = widgets.HTML(
        "<i>No file loaded</i> &nbsp;\u2014&nbsp; "
        f"<small>accepts {', '.join(SUPPORTED_EXT)}</small>"
    )

    # -- 2: config ------------------------------------------------------
    cluster_dd = widgets.Dropdown(
        description="Cluster col", options=[], style={"description_width": "initial"}
    )
    boxplot_ms = widgets.SelectMultiple(
        options=[], rows=8, layout=widgets.Layout(width="260px")
    )
    pct_ms = widgets.SelectMultiple(options=[], rows=8, layout=widgets.Layout(width="260px"))
    sum_ms = widgets.SelectMultiple(options=[], rows=8, layout=widgets.Layout(width="260px"))
    boxplot_color = widgets.ColorPicker(
        value="#7c7c0a", description="Boxplot colour",
        style={"description_width": "initial"},
    )
    sum_color = widgets.ColorPicker(
        value="#e7969c", description="Sum plot colour",
        style={"description_width": "initial"},
    )
    random_cb = widgets.Checkbox(value=False, description="Use random clusters")
    save_cb = widgets.Checkbox(value=False, description="Save plots as PNG")

    scale_container = widgets.VBox()
    scale_widgets = {}

    run_btn = widgets.Button(
        description="Run Cluster Validation",
        button_style="success",
        disabled=True,
        layout=widgets.Layout(width="240px"),
    )
    out = widgets.Output()

    # -- behaviour ------------------------------------------------------
    def rebuild_scales(*_):
        rows = [widgets.HTML("<b>Boxplot Y-axis scale</b>")]
        new = {}
        for col in boxplot_ms.value:
            prev = scale_widgets[col].value if col in scale_widgets else "linear"
            dd = widgets.Dropdown(
                options=SCALE_OPTIONS,
                value=prev,
                description=col,
                style={"description_width": "initial"},
                layout=widgets.Layout(width="320px"),
            )
            new[col] = dd
            rows.append(dd)
        scale_widgets.clear()
        scale_widgets.update(new)
        scale_container.children = tuple(rows) if len(rows) > 1 else ()

    boxplot_ms.observe(rebuild_scales, names="value")

    def _ingest(gdf):
        numeric_cols = [
            c
            for c in gdf.select_dtypes(include="number").columns
            if c not in EXCLUDE_COLS
        ]
        all_cols = [c for c in gdf.columns if c not in {"geometry", "field_1"}]
        state["gdf"] = gdf

        cluster_dd.options = all_cols
        cluster_dd.value = next(
            (c for c in DEFAULTS["cluster_col"] if c in all_cols),
            all_cols[0] if all_cols else None,
        )
        for ms, key in ((boxplot_ms, "boxplot"), (pct_ms, "pct"), (sum_ms, "sum")):
            ms.options = numeric_cols
            ms.value = tuple(c for c in DEFAULTS[key] if c in numeric_cols)
        rebuild_scales()
        run_btn.disabled = False
        geo_note = ""
        if gpd is not None and isinstance(gdf, gpd.GeoDataFrame):
            try:
                kinds = ", ".join(sorted(gdf.geom_type.dropna().unique()))
            except Exception:  # noqa: BLE001
                kinds = "geometry"
            crs = gdf.crs.to_string() if gdf.crs else "no CRS"
            geo_note = f" &nbsp;\u2022&nbsp; {kinds} ({crs})"
        load_status.value = (
            f"<span style='color:#16a34a'>\u2713 Loaded {len(gdf)} rows, "
            f"{len(all_cols)} columns</span>{geo_note}"
        )

    def _current_source():
        """Return the active source: an upload tuple, a path string, or None."""
        item = _upload_item(file_upload)
        if item is not None:
            return item
        if path_box.value.strip():
            return path_box.value.strip()
        return None

    def _refresh_layers(*_):
        """Show a layer picker when the source is a multi-layer geo file."""
        src = _current_source()
        layers = []
        if src is not None:
            name = src[0] if isinstance(src, tuple) else src
            if Path(name).suffix.lower() in {".gpkg", ".gml", ".kml"}:
                probe = src
                # list_layers needs a path; write uploads to a temp file to probe
                if isinstance(src, tuple):
                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=Path(name).suffix.lower(), delete=False
                        ) as tmp:
                            tmp.write(src[1])
                            probe = tmp.name
                        layers = list_layers(probe)
                    finally:
                        try:
                            Path(probe).unlink()
                        except OSError:
                            pass
                else:
                    layers = list_layers(src)
        if len(layers) > 1:
            layer_dd.options = layers
            layer_dd.value = layers[0]
            layer_dd.layout.display = ""
        else:
            layer_dd.options = []
            layer_dd.layout.display = "none"

    file_upload.observe(_refresh_layers, names="value")
    path_box.observe(_refresh_layers, names="value")

    def _on_load(_):
        load_status.value = "<i>Loading\u2026</i>"
        try:
            src = _current_source()
            if src is None:
                load_status.value = (
                    "<span style='color:#dc2626'>\u2717 Choose a file or enter a data "
                    "path</span>"
                )
                return
            layer = layer_dd.value if layer_dd.layout.display != "none" else None
            gdf = read_geodata(src, layer=layer)
            _ingest(gdf)
        except Exception as e:  # noqa: BLE001
            load_status.value = f"<span style='color:#dc2626'>\u2717 {e}</span>"

    def _on_run(_):
        with out:
            clear_output(wait=True)
            gdf = state["gdf"]
            if gdf is None:
                print("No data loaded.")
                return
            scale_map = {c: w.value for c, w in scale_widgets.items()}
            print("Running cluster_validation\u2026")
            print(f"  cluster_col:  {cluster_dd.value}")
            print(f"  boxplot_cols: {list(boxplot_ms.value)}")
            print(f"  pct_cols:     {list(pct_ms.value)}")
            print(f"  sum_cols:     {list(sum_ms.value)}")
            print(f"  yscale map:   {scale_map}")
            print("\u2500" * 50)
            try:
                figures = cluster_validation(
                    gdf=gdf,
                    boxplot_columns=list(boxplot_ms.value),
                    percentage_plot_columns=list(pct_ms.value),
                    sum_plot_columns=list(sum_ms.value),
                    bin_to_label=None,
                    cluster_col=cluster_dd.value,
                    if_random=random_cb.value,
                    boxplot_yscale=scale_map,
                    boxplot_color=boxplot_color.value,
                    sumplot_color=sum_color.value,
                )
            except Exception as e:  # noqa: BLE001
                print(f"ERROR: {e}")
                return

            if save_cb.value and figures:
                OUT_DIR.mkdir(exist_ok=True)
            for i, (t, fig) in enumerate(figures, 1):
                print(t)
                if save_cb.value:
                    safe = re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")
                    p = OUT_DIR / f"plot_{i}_{safe}.png"
                    fig.savefig(p, dpi=150, bbox_inches="tight")
                    print(f"  saved \u2192 {p}")
                display(fig)
                plt.close(fig)
            print("Done!")

    load_btn.on_click(_on_load)
    run_btn.on_click(_on_run)

    # -- layout -------------------------------------------------------------
    load_section = _section(
        "1 \u2014 Load data",
        widgets.HBox([file_upload, load_btn]),
        path_box,
        layer_dd,
        load_status,
    )
    cols_row = widgets.HBox(
        [
            widgets.VBox([widgets.HTML("<b>Boxplot columns</b>"), boxplot_ms]),
            widgets.VBox([widgets.HTML("<b>Percentage plot columns</b>"), pct_ms]),
            widgets.VBox([widgets.HTML("<b>Sum plot columns</b>"), sum_ms]),
        ]
    )
    config_section = _section(
        "2 \u2014 Configure Parameters",
        cluster_dd,
        cols_row,
        scale_container,
        widgets.HBox([boxplot_color, sum_color]),
        random_cb,
        save_cb,
    )
    run_section = _section("3 \u2014 Run", run_btn)

    return widgets.VBox([title, load_section, config_section, run_section, out])


def launch():
    """Build the app and display it."""
    app = build_cluster_validation_app()
    display(app)
    return app
