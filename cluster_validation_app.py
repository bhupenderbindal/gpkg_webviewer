"""Cluster Validation GUI app.

Jupyter reproduction of the ``cluster-vis-py-index.html`` interface, extended
to load geospatial formats (GeoPackage, GeoJSON, Shapefile, FlatGeobuf,
GeoParquet, ...) in addition to plain CSV. Geometry is ignored by the plots;
only the attribute columns are used.

Usage (in a notebook cell)::

    from cluster_validation_app import launch
    launch()

Plotting and data loading live in ``cluster_core`` (shared with the
Streamlit front-end, ``streamlit_app.py``); this module is only the
ipywidgets UI. Dependencies assumed available: numpy, pandas, matplotlib,
seaborn, ipywidgets, geopandas (for .gpkg / .geojson / .shp / ...).
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import clear_output, display

from cluster_core import (  # noqa: F401 - re-exported for existing imports
    DEFAULTS,
    EXCLUDE_COLS,
    OUT_DIR,
    SCALE_OPTIONS,
    SUPPORTED_EXT,
    UPLOAD_ACCEPT,
    cluster_validation,
    gpd,
    list_layers,
    read_geodata,
)


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
    preview_out = widgets.Output(
        layout=widgets.Layout(max_height="420px", overflow="auto")
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
    hue_dd = widgets.Dropdown(
        description="Boxplot hue",
        options=[("(none)", None)],
        value=None,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="320px"),
    )
    rotation_sl = widgets.IntSlider(
        value=0, min=-90, max=90, step=5,
        description="X-label rotation (°)",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="360px"),
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
        # Percentage plots count categories, so string columns are valid there.
        category_cols = [c for c in all_cols if c not in EXCLUDE_COLS]
        for ms, key, opts in (
            (boxplot_ms, "boxplot", numeric_cols),
            (pct_ms, "pct", category_cols),
            (sum_ms, "sum", numeric_cols),
        ):
            ms.options = opts
            ms.value = tuple(c for c in DEFAULTS[key] if c in opts)
        hue_dd.options = [("(none)", None)] + [(c, c) for c in category_cols]
        hue_dd.value = next((c for c in DEFAULTS["hue"] if c in category_cols), None)
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
        _show_preview(gdf)

    def _show_preview(gdf):
        """Render column dtypes and the first 5 rows below the load status."""
        with preview_out:
            clear_output(wait=True)
            dtypes = pd.DataFrame(
                {"column": gdf.columns, "dtype": gdf.dtypes.astype(str).values}
            )
            display(widgets.HTML("<b>Columns &amp; dtypes</b>"))
            display(dtypes.style.hide(axis="index"))
            display(widgets.HTML("<b>First 5 rows</b>"))
            display(pd.DataFrame(gdf.head(5)))

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
        preview_out.clear_output()
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
            print(f"  boxplot hue:  {hue_dd.value}")
            if hue_dd.value is not None and gdf[hue_dd.value].nunique() > 20:
                print(
                    f"  note: hue '{hue_dd.value}' has "
                    f"{gdf[hue_dd.value].nunique()} distinct values — "
                    "boxplots will be crowded; pick a categorical column"
                )
            for label, ms in (
                ("boxplot", boxplot_ms),
                ("percentage plot", pct_ms),
                ("sum plot", sum_ms),
            ):
                if not ms.value:
                    print(f"  note: no {label} columns selected \u2014 skipping")
            for c in pct_ms.value:
                n = gdf[c].nunique()
                if n > 20:
                    print(
                        f"  note: '{c}' has {n} distinct values \u2014 percentage "
                        "plot will be crowded; pick a categorical column"
                    )
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
                    boxplot_hue=hue_dd.value,
                    xlabel_rotation=rotation_sl.value,
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
        preview_out,
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
        widgets.HBox([boxplot_color, hue_dd, sum_color]),
        rotation_sl,
        random_cb,
        save_cb,
    )
    run_section = _section("3 \u2014 Run", run_btn)

    return widgets.VBox([title, load_section, config_section, run_section, out])


def launch():
    """Build the app and display it."""
    app = build_cluster_validation_app()
    # display(app)
    return app
