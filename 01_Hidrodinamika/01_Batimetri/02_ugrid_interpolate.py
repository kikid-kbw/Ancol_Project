"""
UGRID bathymetry interpolation tools

Berisi 2 mode kerja:
1) interpolate  : interpolasi XYZ bathymetry ke UGRID node -> export NC baru
2) qc           : plot dan cek statistik dari NC hasil

Cocok untuk mesh seperti:
- mesh2d_node_x
- mesh2d_node_y
- mesh2d_node_z

Dependensi:
    pip install numpy pandas xarray scipy matplotlib

Cara pakai paling mudah:
- edit bagian USER SETTINGS
- jalankan file ini

Catatan:
- Script ini mengisi bathymetry ke variable mesh2d_node_z
- CRS mesh dan XYZ harus sama
- Z diasumsikan sudah benar tandanya (negatif ke bawah bila itu konvensi data Anda)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree


# ============================================================
# USER SETTINGS
# ============================================================

MODE = "qc"   # pilih: "interpolate" atau "qc"

# ---------- STEP 02: interpolasi ----------
UGRID_INPUT = r"D:\ANCOL\Model\01_Hidrodinamika\01_Batimetri\L0master_v08-ref05_net.nc"
XYZ_INPUT = r"D:\ANCOL\Model\01_Hidrodinamika\01_Batimetri\EDIT_Batimetri-minus_Ancol_UTM_16MAR26.xyz"
NC_OUTPUT = r"D:\ANCOL\Model\01_Hidrodinamika\01_Batimetri\L0_interp_idw_k4_rad250_net.nc"

XYZ_DELIMITER = None   # None = auto whitespace; contoh:
# XYZ_DELIMITER = ","
XYZ_HAS_HEADER = False
XYZ_COLUMNS = ["x", "y", "z"]

INTERP_METHOD = "idw"   # pilihan: "idw", "nearest", "linear"

# parameter IDW
IDW_K = 4
IDW_POWER = 2.0
IDW_RADIUS = 250   # meter. contoh: 500.0 ; None = tanpa radius limit
IDW_FALLBACK_NEAREST = True

# quality control sebelum interpolasi
DROP_DUPLICATE_XY = True
REMOVE_NAN_ROWS = True
Z_MIN = None       # contoh: -50.0
Z_MAX = None       # contoh: 5.0

# output metadata
OUTPUT_Z_VAR_NAME = "mesh2d_node_z"
OUTPUT_TITLE = "UGRID with interpolated bathymetry"
OUTPUT_HISTORY_NOTE = "Bathymetry interpolated from XYZ to mesh2d nodes"

# ---------- STEP 03: QC plot + statistik ----------
QC_NC_INPUT = r"D:\ANCOL\Model\01_Hidrodinamika\01_Batimetri\L0_interp_idw_k4_rad250_net.nc"
QC_SAVE_FIG = True
QC_FIG_PATH = r"D:\ANCOL\Model\01_Hidrodinamika\01_Batimetri\qc_L0_interp_idw_k4_rad250_net.png"
QC_POINT_SIZE = 3
QC_FIG_DPI = 450 #default 200
QC_SHOW_FIG = True

# optional overlay XYZ sumber saat QC
QC_OVERLAY_XYZ = True
QC_XYZ_INPUT = XYZ_INPUT
QC_XYZ_DELIMITER = XYZ_DELIMITER
QC_XYZ_HAS_HEADER = XYZ_HAS_HEADER


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class XYZData:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


# ============================================================
# IO HELPERS
# ============================================================

def read_xyz(
    path: str,
    delimiter: Optional[str] = None,
    has_header: bool = False,
    columns: Optional[list[str]] = None,
) -> XYZData:
    """Read XYZ file into arrays."""
    if columns is None:
        columns = ["x", "y", "z"]

    if has_header:
        df = pd.read_csv(path, sep=delimiter)
        if len(df.columns) >= 3:
            df = df.iloc[:, :3]
            df.columns = columns
        else:
            raise ValueError("Header ditemukan tetapi kolom kurang dari 3.")
    else:
        sep = delimiter if delimiter is not None else r"\s+"
        df = pd.read_csv(path, sep=sep, header=None, names=columns, engine="python")

    return XYZData(
        x=df[columns[0]].to_numpy(dtype=float),
        y=df[columns[1]].to_numpy(dtype=float),
        z=df[columns[2]].to_numpy(dtype=float),
    )


def preprocess_xyz(
    xyz: XYZData,
    remove_nan_rows: bool = True,
    drop_duplicate_xy: bool = True,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
) -> XYZData:
    """Clean XYZ data."""
    df = pd.DataFrame({"x": xyz.x, "y": xyz.y, "z": xyz.z})

    if remove_nan_rows:
        df = df.dropna(subset=["x", "y", "z"])

    if z_min is not None:
        df = df[df["z"] >= z_min]

    if z_max is not None:
        df = df[df["z"] <= z_max]

    if drop_duplicate_xy:
        df = df.groupby(["x", "y"], as_index=False)["z"].mean()

    return XYZData(
        x=df["x"].to_numpy(dtype=float),
        y=df["y"].to_numpy(dtype=float),
        z=df["z"].to_numpy(dtype=float),
    )


def open_ugrid(path: str) -> xr.Dataset:
    ds = xr.open_dataset(path)
    required = ["mesh2d_node_x", "mesh2d_node_y", "mesh2d_node_z"]
    missing = [v for v in required if v not in ds.variables]
    if missing:
        raise KeyError(f"Variabel wajib tidak ditemukan di UGRID: {missing}")
    return ds


# ============================================================
# INTERPOLATION HELPERS
# ============================================================

def interpolate_griddata(
    src_x: np.ndarray,
    src_y: np.ndarray,
    src_z: np.ndarray,
    trg_x: np.ndarray,
    trg_y: np.ndarray,
    method: str,
) -> np.ndarray:
    points = np.column_stack([src_x, src_y])
    targets = (trg_x, trg_y)
    zi = griddata(points, src_z, targets, method=method)

    # fallback untuk area di luar convex hull saat linear
    if method == "linear":
        mask = np.isnan(zi)
        if np.any(mask):
            zi[mask] = griddata(points, src_z, (trg_x[mask], trg_y[mask]), method="nearest")
    return zi


def interpolate_idw(
    src_x: np.ndarray,
    src_y: np.ndarray,
    src_z: np.ndarray,
    trg_x: np.ndarray,
    trg_y: np.ndarray,
    k: int = 8,
    power: float = 2.0,
    radius: Optional[float] = None,
    fallback_nearest: bool = True,
) -> np.ndarray:
    src_xy = np.column_stack([src_x, src_y])
    trg_xy = np.column_stack([trg_x, trg_y])

    tree = cKDTree(src_xy)

    if radius is None:
        dist, idx = tree.query(trg_xy, k=k)
    else:
        dist, idx = tree.query(trg_xy, k=k, distance_upper_bound=radius)

    # pastikan 2D bila k=1
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    zi = np.full(len(trg_xy), np.nan, dtype=float)
    src_n = len(src_z)
    eps = 1.0e-12

    valid = idx < src_n
    any_valid = valid.any(axis=1)

    for i in np.where(any_valid)[0]:
        d = dist[i, valid[i]]
        j = idx[i, valid[i]]

        # kalau ada titik persis sama
        zero_mask = d <= eps
        if np.any(zero_mask):
            zi[i] = src_z[j[zero_mask][0]]
            continue

        w = 1.0 / np.power(np.maximum(d, eps), power)
        zi[i] = np.sum(w * src_z[j]) / np.sum(w)

    if fallback_nearest and np.any(np.isnan(zi)):
        nan_mask = np.isnan(zi)
        d1, j1 = tree.query(trg_xy[nan_mask], k=1)
        zi[nan_mask] = src_z[j1]

    return zi


# ============================================================
# QC / STATS
# ============================================================

def describe_array(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=float)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"count      : {arr.size}")
    print(f"nan_count  : {np.isnan(arr).sum()}")
    print(f"min        : {np.nanmin(arr):.6f}")
    print(f"max        : {np.nanmax(arr):.6f}")
    print(f"mean       : {np.nanmean(arr):.6f}")
    print(f"std        : {np.nanstd(arr):.6f}")
    print(f"p01        : {np.nanpercentile(arr, 1):.6f}")
    print(f"p05        : {np.nanpercentile(arr, 5):.6f}")
    print(f"p50        : {np.nanpercentile(arr, 50):.6f}")
    print(f"p95        : {np.nanpercentile(arr, 95):.6f}")
    print(f"p99        : {np.nanpercentile(arr, 99):.6f}")


def plot_bathy_nodes(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    point_size: float = 3,
    save_fig: bool = False,
    fig_path: Optional[str] = None,
    dpi: int = 200,
    show_fig: bool = True,
    overlay_xyz: Optional[XYZData] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(x, y, c=z, s=point_size)
    plt.colorbar(sc, ax=ax, label="Bed level (m)")

    if overlay_xyz is not None:
        ax.scatter(overlay_xyz.x, overlay_xyz.y, s=0.3, c="k", alpha=0.15, label="Source XYZ")
        ax.legend(loc="best")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Bathymetry on UGRID nodes")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    if save_fig and fig_path:
        out_dir = os.path.dirname(fig_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        print(f"\nFigure saved: {fig_path}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# MAIN WORKFLOWS
# ============================================================

def run_interpolation() -> None:
    print("Membuka UGRID...")
    ds = open_ugrid(UGRID_INPUT)

    print("Membaca XYZ...")
    xyz_raw = read_xyz(
        path=XYZ_INPUT,
        delimiter=XYZ_DELIMITER,
        has_header=XYZ_HAS_HEADER,
        columns=XYZ_COLUMNS,
    )

    print("Preprocess XYZ...")
    xyz = preprocess_xyz(
        xyz=xyz_raw,
        remove_nan_rows=REMOVE_NAN_ROWS,
        drop_duplicate_xy=DROP_DUPLICATE_XY,
        z_min=Z_MIN,
        z_max=Z_MAX,
    )

    print("Statistik XYZ sumber:")
    describe_array("XYZ source z", xyz.z)

    node_x = np.asarray(ds["mesh2d_node_x"].values, dtype=float)
    node_y = np.asarray(ds["mesh2d_node_y"].values, dtype=float)

    print(f"\nJumlah node target: {len(node_x)}")
    print(f"Metode interpolasi: {INTERP_METHOD}")

    if INTERP_METHOD.lower() in ["nearest", "linear"]:
        z_new = interpolate_griddata(
            src_x=xyz.x,
            src_y=xyz.y,
            src_z=xyz.z,
            trg_x=node_x,
            trg_y=node_y,
            method=INTERP_METHOD.lower(),
        )
    elif INTERP_METHOD.lower() == "idw":
        z_new = interpolate_idw(
            src_x=xyz.x,
            src_y=xyz.y,
            src_z=xyz.z,
            trg_x=node_x,
            trg_y=node_y,
            k=IDW_K,
            power=IDW_POWER,
            radius=IDW_RADIUS,
            fallback_nearest=IDW_FALLBACK_NEAREST,
        )
    else:
        raise ValueError("INTERP_METHOD harus salah satu: 'idw', 'nearest', 'linear'")

    print("Statistik hasil interpolasi:")
    describe_array("Interpolated node z", z_new)

    print("Menyalin dataset dan mengganti mesh2d_node_z...")
    ds_out = ds.copy(deep=True)
    ds_out[OUTPUT_Z_VAR_NAME] = (("mesh2d_nNodes",), z_new)

    # jaga atribut lama jika ada
    old_attrs = ds["mesh2d_node_z"].attrs.copy() if "mesh2d_node_z" in ds.variables else {}
    old_attrs.update({
        "long_name": "Interpolated bathymetry at mesh nodes",
        "units": "m",
        "mesh": "mesh2d",
        "location": "node",
        "grid_mapping": "projected_coordinate_system",
    })
    ds_out[OUTPUT_Z_VAR_NAME].attrs = old_attrs

    ds_out.attrs = ds.attrs.copy()
    ds_out.attrs["title"] = OUTPUT_TITLE
    hist_old = ds.attrs.get("history", "")
    hist_new = f"{hist_old} | {OUTPUT_HISTORY_NOTE}"
    ds_out.attrs["history"] = hist_new

    out_dir = os.path.dirname(NC_OUTPUT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"Menyimpan NC baru: {NC_OUTPUT}")
    ds_out.to_netcdf(NC_OUTPUT)
    ds.close()
    ds_out.close()
    print("Selesai.")


def run_qc() -> None:
    print(f"Membuka file QC: {QC_NC_INPUT}")
    ds = xr.open_dataset(QC_NC_INPUT)

    required = ["mesh2d_node_x", "mesh2d_node_y", "mesh2d_node_z"]
    missing = [v for v in required if v not in ds.variables]
    if missing:
        raise KeyError(f"Variabel wajib tidak ditemukan di QC file: {missing}")

    x = np.asarray(ds["mesh2d_node_x"].values, dtype=float)
    y = np.asarray(ds["mesh2d_node_y"].values, dtype=float)
    z = np.asarray(ds["mesh2d_node_z"].values, dtype=float)

    print("Informasi dataset:")
    print(ds)
    describe_array("QC mesh2d_node_z", z)

    overlay = None
    if QC_OVERLAY_XYZ and os.path.exists(QC_XYZ_INPUT):
        xyz = read_xyz(
            path=QC_XYZ_INPUT,
            delimiter=QC_XYZ_DELIMITER,
            has_header=QC_XYZ_HAS_HEADER,
            columns=XYZ_COLUMNS,
        )
        overlay = preprocess_xyz(
            xyz=xyz,
            remove_nan_rows=True,
            drop_duplicate_xy=False,
            z_min=None,
            z_max=None,
        )

    plot_bathy_nodes(
        x=x,
        y=y,
        z=z,
        point_size=QC_POINT_SIZE,
        save_fig=QC_SAVE_FIG,
        fig_path=QC_FIG_PATH,
        dpi=QC_FIG_DPI,
        show_fig=QC_SHOW_FIG,
        overlay_xyz=overlay,
    )

    ds.close()
    print("QC selesai.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    if MODE.lower() == "interpolate":
        run_interpolation()
    elif MODE.lower() == "qc":
        run_qc()
    else:
        raise ValueError("MODE harus 'interpolate' atau 'qc'")
