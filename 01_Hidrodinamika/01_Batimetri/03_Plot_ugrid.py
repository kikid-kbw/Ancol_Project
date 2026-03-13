"""
03_plot_ugrid_grid.py

Plot bathymetry hasil interpolasi pada UGRID netCDF
tanpa overlay XYZ, dengan garis grid/mesh.

Dependensi:
    pip install numpy xarray matplotlib

Catatan:
- Script ini membaca:
    mesh2d_node_x
    mesh2d_node_y
    mesh2d_node_z
    mesh2d_edge_nodes
- mesh2d_edge_nodes memakai start_index = 1, jadi dikonversi ke index Python
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# =========================================================
# USER SETTINGS
# =========================================================

NC_INPUT = r"D:\ANCOL\Model\01_Hidrodinamika\01_Batimetri\L0_interp_idw_k4_rad250_net.nc"
FIG_OUTPUT = r"D:\ANCOL\Model\01_Hidrodinamika\01_Batimetri\plot_grid_L0_interp_idw_k4_rad250_net.png"

SHOW_FIG = True
SAVE_FIG = True
FIG_DPI = 250

# tampilan
FIGSIZE = (10, 8)
NODE_SIZE = 8
EDGE_LINEWIDTH = 0.25
EDGE_ALPHA = 0.5
EDGE_COLOR = "k"
TITLE = "Bathymetry on UGRID nodes with mesh lines"

# color scale
# isi manual bila ingin fix, mis. VMIN=-10, VMAX=0
# kalau None, otomatis dari area aktif (full domain atau area zoom)
VMIN = None
VMAX = None

# zoom area
USE_ZOOM = True
XMIN = 700756.799
XMAX = 706887.171
YMIN = 9322738.523
YMAX = 9325735.593


# =========================================================
# FUNCTIONS
# =========================================================

def open_ugrid(nc_path):
    ds = xr.open_dataset(nc_path)

    required = [
        "mesh2d_node_x",
        "mesh2d_node_y",
        "mesh2d_node_z",
        "mesh2d_edge_nodes",
    ]
    missing = [v for v in required if v not in ds.variables]
    if missing:
        raise KeyError(f"Variabel wajib tidak ditemukan: {missing}")

    node_x = np.asarray(ds["mesh2d_node_x"].values, dtype=float)
    node_y = np.asarray(ds["mesh2d_node_y"].values, dtype=float)
    node_z = np.asarray(ds["mesh2d_node_z"].values, dtype=float)
    edge_nodes = np.asarray(ds["mesh2d_edge_nodes"].values)

    start_index = ds["mesh2d_edge_nodes"].attrs.get("start_index", 0)

    return ds, node_x, node_y, node_z, edge_nodes, start_index


def convert_edge_index_to_python(edge_nodes, start_index):
    edges = np.asarray(edge_nodes, dtype=int).copy()

    if start_index == 1:
        edges = edges - 1

    return edges


def get_zoom_mask(node_x, node_y, use_zoom=False, xmin=None, xmax=None, ymin=None, ymax=None):
    if not use_zoom:
        return np.ones(len(node_x), dtype=bool)

    return (
        (node_x >= xmin) & (node_x <= xmax) &
        (node_y >= ymin) & (node_y <= ymax)
    )


def print_stats(name, arr):
    arr = np.asarray(arr, dtype=float)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"count     : {arr.size}")
    print(f"nan_count : {np.isnan(arr).sum()}")
    print(f"min       : {np.nanmin(arr):.6f}")
    print(f"max       : {np.nanmax(arr):.6f}")
    print(f"mean      : {np.nanmean(arr):.6f}")
    print(f"std       : {np.nanstd(arr):.6f}")


def plot_ugrid_with_edges(
    node_x,
    node_y,
    node_z,
    edges,
    fig_output=None,
    show_fig=True,
    save_fig=True,
    fig_dpi=250,
    figsize=(10, 8),
    node_size=8,
    edge_linewidth=0.25,
    edge_alpha=0.5,
    edge_color="k",
    title="Bathymetry on UGRID nodes with mesh lines",
    vmin=None,
    vmax=None,
    use_zoom=False,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
):
    zoom_mask = get_zoom_mask(
        node_x=node_x,
        node_y=node_y,
        use_zoom=use_zoom,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
    )

    if not np.any(zoom_mask):
        raise ValueError("Area zoom tidak memotong node mana pun. Cek XMIN/XMAX/YMIN/YMAX.")

    # data aktif untuk scatter dan autoscale colorbar
    x_plot = node_x[zoom_mask]
    y_plot = node_y[zoom_mask]
    z_plot = node_z[zoom_mask]

    # autoscale colorbar berdasarkan area aktif
    if vmin is None:
        vmin = np.nanmin(z_plot)
    if vmax is None:
        vmax = np.nanmax(z_plot)

    fig, ax = plt.subplots(figsize=figsize)

    sc = ax.scatter(
        x_plot,
        y_plot,
        c=z_plot,
        s=node_size,
        vmin=vmin,
        vmax=vmax,
    )

    # plot edge mesh
    for n1, n2 in edges:
        if n1 < 0 or n2 < 0:
            continue
        if n1 >= len(node_x) or n2 >= len(node_x):
            continue

        if use_zoom:
            in1 = zoom_mask[n1]
            in2 = zoom_mask[n2]
            if not (in1 or in2):
                continue

        ax.plot(
            [node_x[n1], node_x[n2]],
            [node_y[n1], node_y[n2]],
            linewidth=edge_linewidth,
            alpha=edge_alpha,
            color=edge_color,
        )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Bed level (m)")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)

    if use_zoom:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    if save_fig and fig_output:
        outdir = os.path.dirname(fig_output)
        if outdir:
            os.makedirs(outdir, exist_ok=True)
        plt.savefig(fig_output, dpi=fig_dpi, bbox_inches="tight")
        print(f"\nFigure saved: {fig_output}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    print(f"Membuka file: {NC_INPUT}")
    ds, node_x, node_y, node_z, edge_nodes, start_index = open_ugrid(NC_INPUT)

    print("Informasi dataset:")
    print(ds)

    print_stats("Statistik mesh2d_node_z (full domain)", node_z)

    zoom_mask = get_zoom_mask(
        node_x=node_x,
        node_y=node_y,
        use_zoom=USE_ZOOM,
        xmin=XMIN,
        xmax=XMAX,
        ymin=YMIN,
        ymax=YMAX,
    )

    if np.any(zoom_mask):
        print_stats("Statistik mesh2d_node_z (active view)", node_z[zoom_mask])

    edges = convert_edge_index_to_python(edge_nodes, start_index)

    plot_ugrid_with_edges(
        node_x=node_x,
        node_y=node_y,
        node_z=node_z,
        edges=edges,
        fig_output=FIG_OUTPUT,
        show_fig=SHOW_FIG,
        save_fig=SAVE_FIG,
        fig_dpi=FIG_DPI,
        figsize=FIGSIZE,
        node_size=NODE_SIZE,
        edge_linewidth=EDGE_LINEWIDTH,
        edge_alpha=EDGE_ALPHA,
        edge_color=EDGE_COLOR,
        title=TITLE,
        vmin=VMIN,
        vmax=VMAX,
        use_zoom=USE_ZOOM,
        xmin=XMIN,
        xmax=XMAX,
        ymin=YMIN,
        ymax=YMAX,
    )

    ds.close()
    print("Selesai.")