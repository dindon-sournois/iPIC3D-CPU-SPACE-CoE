"""
    SCRIPT="../postprocessing_tools/python/B_J.py"
    DATA_DIR="/scratch/project_465003132/test_xy/"
    OUT_DIR="/scratch/project_465003132/test_xy/plots"

    ### 3D run (3 panels Bx(XY), Jz(XY), Jz(ZY))
    srun python3 "$SCRIPT" "$DATA_DIR" \
        $xmin $xmax $ymin $ymax $zmin $zmax \
        --nxc "$nxc" --nyc "$nyc" --nzc "$nzc" \
        --cycle-start 0 --cycle-end 5000 --cycle-step 100 --time-step 100 \
        --outdir "$OUT_DIR"

    ### 2D run (2 panels Bx, Jz on the resolved plane)
    srun python3 "$SCRIPT" "$DATA_DIR" \
        $xmin $xmax $ymin $ymax $zmin $zmax \
        --nxc "$nxc" --nyc "$nyc" --nzc "$nzc" \
        --cycle-start 0 --cycle-end 5000 --cycle-step 100 \
        --outdir "$OUT_DIR" --plane XY
"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MPLBACKEND", "Agg")

import glob
import argparse
from datetime import datetime

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = datetime.now()

comm.Barrier()
if rank == 0:
    print("All ranks passed first barrier", flush=True)


def proc_id_from_filename(fp):
    base = os.path.basename(fp)
    return int(base.replace("proc", "").replace(".hdf", ""))


def mapping_candidates(XLEN, YLEN, ZLEN):
    ###! proc_id -> (i,j,k). Six common orderings; the right one is inferred by
    ###! grid occupancy (a mis-ordering leaves gaps/overlaps in the tile grid).
    def A(pid):
        k = pid % ZLEN
        t = pid // ZLEN
        j = t % YLEN
        i = t // YLEN
        return i, j, k

    def B(pid):
        j = pid % YLEN
        t = pid // YLEN
        k = t % ZLEN
        i = t // ZLEN
        return i, j, k

    def C(pid):
        k = pid % ZLEN
        t = pid // ZLEN
        i = t % XLEN
        j = t // XLEN
        return i, j, k

    def D(pid):
        i = pid % XLEN
        t = pid // XLEN
        j = t % YLEN
        k = t // YLEN
        return i, j, k

    def E(pid):
        j = pid % YLEN
        t = pid // YLEN
        i = t % XLEN
        k = t // XLEN
        return i, j, k

    def F(pid):
        i = pid % XLEN
        t = pid // XLEN
        k = t % ZLEN
        j = t // ZLEN
        return i, j, k

    return {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}


def choose_mapping(files, XLEN, YLEN, ZLEN):
    ###! Score each candidate by grid occupancy: the correct proc -> (i,j,k)
    ###! covers every tile exactly once (score 0). Gaps/overlaps are penalised.
    ###! When LEN factors share divisors several candidates can tie at 0; this
    ###! returns the first best. If a run is ambiguous, pass --mapping.
    proc_ids = [proc_id_from_filename(fp) for fp in files]
    maps = mapping_candidates(XLEN, YLEN, ZLEN)

    best_name = None
    best_score = None

    for name, fn in maps.items():
        occ = np.zeros((XLEN, YLEN, ZLEN), dtype=np.int32)
        valid = True

        for pid in proc_ids:
            i, j, k = fn(pid)

            if not (0 <= i < XLEN and 0 <= j < YLEN and 0 <= k < ZLEN):
                valid = False
                break

            occ[i, j, k] += 1

        if not valid:
            continue

        gaps = int(np.count_nonzero(occ == 0))
        overlaps = int(np.count_nonzero(occ > 1))
        score = 10 * gaps + 100 * overlaps

        if best_score is None or score < best_score:
            best_score = score
            best_name = name

    if best_name is None:
        raise RuntimeError("Could not determine proc -> (i,j,k) mapping.")

    return best_name, best_score


def global_shape_shared(tile_shape, XLEN, YLEN, ZLEN):
    ###! Shared-boundary assembly: neighbouring tiles share one boundary plane,
    ###! so global nodes = LEN*(n_tile - 1) + 1, NOT LEN*n_tile. On a collapsed
    ###! 2D axis (LEN = 1, n_tile = 2) this gives 1*(2-1)+1 = 2 nodes.
    nx, ny, nz = tile_shape
    nx_global = XLEN * (nx - 1) + 1
    ny_global = YLEN * (ny - 1) + 1
    nz_global = ZLEN * (nz - 1) + 1
    return nx_global, ny_global, nz_global


def global_offset_shared(i, j, k, tile_shape):
    nx, ny, nz = tile_shape
    x0 = i * (nx - 1)
    y0 = j * (ny - 1)
    z0 = k * (nz - 1)
    return x0, y0, z0


def lower_crop_indices(i, j, k):
    ###! Every tile but the first along an axis repeats its lower boundary plane
    ###! (shared with the previous tile); drop it to avoid double-writing.
    xs = 0 if i == 0 else 1
    ys = 0 if j == 0 else 1
    zs = 0 if k == 0 else 1
    return xs, ys, zs


def jz_path(current_mode, sp, cycle_name):
    ###! total mode: single combined dataset; species mode: per-species dataset
    if current_mode == "total":
        return f"moments/Jz/{cycle_name}"
    return f"moments/species_{sp}/Jz/{cycle_name}"


###! ============================================================
###! 3D assembly (ORIGINAL behaviour): Bx(XY), Jz(XY) at z=z_index and
###! Jz(ZY) at x=x_index, all on the global grid.
###! ============================================================

def read_current_sum_xy(f, cycle_name, species_list, raw_k, xs, ys, current_mode):
    ###! total -> one dataset (sp ignored); species -> sum over species_list
    sp_iter = [None] if current_mode == "total" else species_list
    Jz_sum = None

    for sp in sp_iter:
        path = jz_path(current_mode, sp, cycle_name)

        if path not in f:
            raise KeyError(f"Missing dataset: {path}")

        slab = np.array(f[path][xs:, ys:, raw_k], dtype=np.float64)

        if Jz_sum is None:
            Jz_sum = np.zeros_like(slab, dtype=np.float64)

        Jz_sum += slab

    return Jz_sum


def read_current_sum_zy(f, cycle_name, species_list, raw_i, ys, zs, current_mode):
    sp_iter = [None] if current_mode == "total" else species_list
    Jz_sum = None

    for sp in sp_iter:
        path = jz_path(current_mode, sp, cycle_name)

        if path not in f:
            raise KeyError(f"Missing dataset: {path}")

        slab_yz = np.array(f[path][raw_i, ys:, zs:], dtype=np.float64)
        slab_zy = slab_yz.T

        if Jz_sum is None:
            Jz_sum = np.zeros_like(slab_zy, dtype=np.float64)

        Jz_sum += slab_zy

    return Jz_sum


def assemble_cycle_3d(cycle_name, local_files, rank_to_ijk, Bx_shape, Jz_shape,
                      Bx_global_shape, Jz_global_shape, z_index, x_index,
                      species_list, current_mode):
    Bx_nx, Bx_ny, Bx_nz = Bx_global_shape
    Jz_nx, Jz_ny, Jz_nz = Jz_global_shape

    local_Bx_XY = np.zeros((Bx_nx, Bx_ny), dtype=np.float64)
    local_Jz_XY = np.zeros((Jz_nx, Jz_ny), dtype=np.float64)
    local_Jz_ZY = np.zeros((Jz_nz, Jz_ny), dtype=np.float64)

    local_Bx_occ = np.zeros((Bx_nx, Bx_ny), dtype=np.int32)
    local_Jz_XY_occ = np.zeros((Jz_nx, Jz_ny), dtype=np.int32)
    local_Jz_ZY_occ = np.zeros((Jz_nz, Jz_ny), dtype=np.int32)

    for fp in local_files:
        pid = proc_id_from_filename(fp)
        i, j, k = rank_to_ijk(pid)

        Bx_raw_nx, Bx_raw_ny, Bx_raw_nz = Bx_shape
        Jz_raw_nx, Jz_raw_ny, Jz_raw_nz = Jz_shape

        Bx_ox, Bx_oy, Bx_oz = global_offset_shared(i, j, k, Bx_shape)
        Jz_ox, Jz_oy, Jz_oz = global_offset_shared(i, j, k, Jz_shape)

        Bx_xs, Bx_ys, Bx_zs = lower_crop_indices(i, j, k)
        Jz_xs, Jz_ys, Jz_zs = lower_crop_indices(i, j, k)

        Bx_x0 = Bx_ox + Bx_xs
        Bx_y0 = Bx_oy + Bx_ys
        Bx_z0 = Bx_oz + Bx_zs

        Jz_x0 = Jz_ox + Jz_xs
        Jz_y0 = Jz_oy + Jz_ys
        Jz_z0 = Jz_oz + Jz_zs

        Bx_use_nx = Bx_raw_nx - Bx_xs
        Bx_use_ny = Bx_raw_ny - Bx_ys
        Bx_use_nz = Bx_raw_nz - Bx_zs

        Jz_use_nx = Jz_raw_nx - Jz_xs
        Jz_use_ny = Jz_raw_ny - Jz_ys
        Jz_use_nz = Jz_raw_nz - Jz_zs

        Bx_has_xy = Bx_z0 <= z_index < Bx_z0 + Bx_use_nz
        Jz_has_xy = Jz_z0 <= z_index < Jz_z0 + Jz_use_nz
        Jz_has_zy = Jz_x0 <= x_index < Jz_x0 + Jz_use_nx

        if not (Bx_has_xy or Jz_has_xy or Jz_has_zy):
            continue

        with h5py.File(fp, "r") as f:
            if Bx_has_xy:
                raw_k = z_index - Bx_oz
                path = f"fields/Bx/{cycle_name}"

                if path not in f:
                    raise KeyError(f"Missing dataset: {path}")

                Bx_slab = np.array(f[path][Bx_xs:, Bx_ys:, raw_k], dtype=np.float64)
                local_Bx_XY[Bx_x0:Bx_x0 + Bx_use_nx, Bx_y0:Bx_y0 + Bx_use_ny] = Bx_slab
                local_Bx_occ[Bx_x0:Bx_x0 + Bx_use_nx, Bx_y0:Bx_y0 + Bx_use_ny] += 1

            if Jz_has_xy:
                raw_k = z_index - Jz_oz
                Jz_slab = read_current_sum_xy(f, cycle_name, species_list, raw_k,
                                              Jz_xs, Jz_ys, current_mode)
                local_Jz_XY[Jz_x0:Jz_x0 + Jz_use_nx, Jz_y0:Jz_y0 + Jz_use_ny] = Jz_slab
                local_Jz_XY_occ[Jz_x0:Jz_x0 + Jz_use_nx, Jz_y0:Jz_y0 + Jz_use_ny] += 1

            if Jz_has_zy:
                raw_i = x_index - Jz_ox
                Jz_slab = read_current_sum_zy(f, cycle_name, species_list, raw_i,
                                              Jz_ys, Jz_zs, current_mode)
                local_Jz_ZY[Jz_z0:Jz_z0 + Jz_use_nz, Jz_y0:Jz_y0 + Jz_use_ny] = Jz_slab
                local_Jz_ZY_occ[Jz_z0:Jz_z0 + Jz_use_nz, Jz_y0:Jz_y0 + Jz_use_ny] += 1

    Bx_XY = None
    Jz_XY = None
    Jz_ZY = None
    Bx_occ = None
    Jz_XY_occ = None
    Jz_ZY_occ = None

    if rank == 0:
        Bx_XY = np.zeros((Bx_nx, Bx_ny), dtype=np.float64)
        Jz_XY = np.zeros((Jz_nx, Jz_ny), dtype=np.float64)
        Jz_ZY = np.zeros((Jz_nz, Jz_ny), dtype=np.float64)
        Bx_occ = np.zeros((Bx_nx, Bx_ny), dtype=np.int32)
        Jz_XY_occ = np.zeros((Jz_nx, Jz_ny), dtype=np.int32)
        Jz_ZY_occ = np.zeros((Jz_nz, Jz_ny), dtype=np.int32)

    comm.Reduce(local_Bx_XY, Bx_XY, op=MPI.SUM, root=0)
    comm.Reduce(local_Jz_XY, Jz_XY, op=MPI.SUM, root=0)
    comm.Reduce(local_Jz_ZY, Jz_ZY, op=MPI.SUM, root=0)

    comm.Reduce(local_Bx_occ, Bx_occ, op=MPI.SUM, root=0)
    comm.Reduce(local_Jz_XY_occ, Jz_XY_occ, op=MPI.SUM, root=0)
    comm.Reduce(local_Jz_ZY_occ, Jz_ZY_occ, op=MPI.SUM, root=0)

    return Bx_XY, Jz_XY, Jz_ZY, Bx_occ, Jz_XY_occ, Jz_ZY_occ


def plot_cycle_3d(cycle_name, Bx_XY, Jz_XY, Jz_ZY, Bx_occ, Jz_XY_occ, Jz_ZY_occ,
                  outdir, extents):
    ###! extents = (xmin, xmax, ymin, ymax, zmin, zmax) in physical units.
    xmin, xmax, ymin, ymax, zmin, zmax = extents

    if Bx_occ.min() == 0:
        print(f"WARNING: {cycle_name} Bx XY has gaps.", flush=True)

    if Jz_XY_occ.min() == 0:
        print(f"WARNING: {cycle_name} Jz XY has gaps.", flush=True)

    if Jz_ZY_occ.min() == 0:
        print(f"WARNING: {cycle_name} Jz ZY has gaps.", flush=True)

    fig, axs = plt.subplots(1, 3, figsize=(16, 8), dpi=200)

    ###! XY planes: horizontal = x (xmin..xmax), vertical = y (ymin..ymax)
    im0 = axs[0].imshow(Bx_XY.T, origin="lower", cmap="seismic", aspect="auto",
                        vmin=-0.25, vmax=0.25, extent=[xmin, xmax, ymin, ymax])
    axs[0].set_title(r"$B_x$ $(X,Y)$", fontsize=16)
    axs[0].set_xlabel(r"$x\,\omega_p/c$", fontsize=12)
    axs[0].set_ylabel(r"$y\,\omega_p/c$", fontsize=12)
    axs[0].tick_params(axis="both", which="major", labelsize=12, length=6)
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(Jz_XY.T, origin="lower", cmap="seismic", aspect="auto",
                        vmin=-0.08, vmax=0.08, extent=[xmin, xmax, ymin, ymax])
    axs[1].set_title(r"$J_z$ $(X,Y)$", fontsize=16)
    axs[1].set_xlabel(r"$x\,\omega_p/c$", fontsize=12)
    axs[1].set_ylabel(r"$y\,\omega_p/c$", fontsize=12)
    axs[1].tick_params(axis="both", which="major", labelsize=12, length=6)
    fig.colorbar(im1, ax=axs[1])

    ###! ZY plane: horizontal = z (zmin..zmax), vertical = y (ymin..ymax)
    im2 = axs[2].imshow(Jz_ZY.T, origin="lower", cmap="seismic", aspect="auto",
                        vmin=-0.08, vmax=0.08, extent=[zmin, zmax, ymin, ymax])
    axs[2].set_title(r"$J_z$ $(Z,Y)$", fontsize=16)
    axs[2].set_xlabel(r"$z\,\omega_p/c$", fontsize=12)
    axs[2].set_ylabel(r"$y\,\omega_p/c$", fontsize=12)
    axs[2].tick_params(axis="both", which="major", labelsize=12, length=6)
    fig.colorbar(im2, ax=axs[2])

    cycle_number = int(cycle_name.replace("cycle_", ""))
    time_omega = cycle_number / args.cycle_step * args.time_step
    fig.suptitle(rf"$T = {time_omega}\,\omega_p^{{-1}}$", fontsize=18)
    fig.tight_layout()

    outfile = os.path.join(outdir, f"{cycle_name}.png")
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {outfile}", flush=True)


###! ============================================================
###! 2D-spatial assembly: Bx and Jz on the single resolved plane. `plane`
###! selects which two axes are resolved and which is collapsed (sampled at
###! its lower global node). Returned arrays are oriented data[H, V]:
###!   XY -> [x, y]    YZ -> [z, y]    ZX -> [z, x]
###! ============================================================

def read_field_plane(f, path, xs, ys, zs, plane, coll_local):
    d = f[path]

    if plane == "XY":
        ###! collapse z: one z node, keep (x, y). Crop x,y shared planes.
        return np.array(d[xs:, ys:, coll_local], dtype=np.float64)          ###! [x, y]

    if plane == "YZ":
        ###! collapse x: one x node, keep (y, z). Want [z, y] -> transpose.
        slab_yz = np.array(d[coll_local, ys:, zs:], dtype=np.float64)       ###! [y, z]
        return slab_yz.T                                                    ###! [z, y]

    if plane == "ZX":
        ###! collapse y: one y node, keep (x, z). Want [z, x] -> transpose.
        slab_xz = np.array(d[xs:, coll_local, zs:], dtype=np.float64)       ###! [x, z]
        return slab_xz.T                                                    ###! [z, x]

    raise ValueError(f"Unknown plane '{plane}'")


def read_current_plane(f, cycle_name, species_list, xs, ys, zs, plane,
                       coll_local, current_mode):
    ###! total -> one dataset (sp ignored); species -> sum over species_list.
    sp_iter = [None] if current_mode == "total" else species_list
    Jz_sum = None

    for sp in sp_iter:
        path = jz_path(current_mode, sp, cycle_name)

        if path not in f:
            raise KeyError(f"Missing dataset: {path}")

        slab = read_field_plane(f, path, xs, ys, zs, plane, coll_local)

        if Jz_sum is None:
            Jz_sum = np.zeros_like(slab, dtype=np.float64)

        Jz_sum += slab

    return Jz_sum


def assemble_cycle_2d(cycle_name, local_files, rank_to_ijk, Bx_shape, Jz_shape,
                      Bx_global_shape, Jz_global_shape, plane, coll_index,
                      species_list, current_mode):
    Bx_nx, Bx_ny, Bx_nz = Bx_global_shape
    Jz_nx, Jz_ny, Jz_nz = Jz_global_shape

    ###! resolved-plane global dims (nH, nV) for Bx and Jz
    if plane == "XY":
        Bx_H, Bx_V = Bx_nx, Bx_ny
        Jz_H, Jz_V = Jz_nx, Jz_ny
    elif plane == "YZ":
        Bx_H, Bx_V = Bx_nz, Bx_ny
        Jz_H, Jz_V = Jz_nz, Jz_ny
    elif plane == "ZX":
        Bx_H, Bx_V = Bx_nz, Bx_nx
        Jz_H, Jz_V = Jz_nz, Jz_nx
    else:
        raise ValueError(f"Unknown plane '{plane}'")

    local_Bx = np.zeros((Bx_H, Bx_V), dtype=np.float64)
    local_Jz = np.zeros((Jz_H, Jz_V), dtype=np.float64)
    local_Bx_occ = np.zeros((Bx_H, Bx_V), dtype=np.int32)
    local_Jz_occ = np.zeros((Jz_H, Jz_V), dtype=np.int32)

    def plane_placement(offs, crops, raw):
        ###! Given a tile's global offsets, shared-plane crops and raw tile
        ###! shape, return its origin (H0,V0) and size (nHu,nVu) on the resolved
        ###! plane, whether it owns the collapsed node `coll_index`, and the
        ###! local index of that node. axes: XY collapse z, YZ collapse x, ZX collapse y.
        ox, oy, oz = offs
        cs = crops
        nx_t, ny_t, nz_t = raw
        if plane == "XY":                             ###! H=x, V=y ; collapse z
            H0 = ox + cs[0]; V0 = oy + cs[1]
            nHu = nx_t - cs[0]; nVu = ny_t - cs[1]
            c_o, c_s, c_n = oz, cs[2], nz_t
        elif plane == "YZ":                           ###! H=z, V=y ; collapse x
            H0 = oz + cs[2]; V0 = oy + cs[1]
            nHu = nz_t - cs[2]; nVu = ny_t - cs[1]
            c_o, c_s, c_n = ox, cs[0], nx_t
        else:                                         ###! ZX: H=z, V=x ; collapse y
            H0 = oz + cs[2]; V0 = ox + cs[0]
            nHu = nz_t - cs[2]; nVu = nx_t - cs[0]
            c_o, c_s, c_n = oy, cs[1], ny_t
        has = (c_o + c_s) <= coll_index < (c_o + c_n)
        coll_local = (coll_index - c_o) if has else None
        return H0, V0, nHu, nVu, has, coll_local

    for fp in local_files:
        pid = proc_id_from_filename(fp)
        i, j, k = rank_to_ijk(pid)

        Bx_offs = global_offset_shared(i, j, k, Bx_shape)
        Jz_offs = global_offset_shared(i, j, k, Jz_shape)
        crops = lower_crop_indices(i, j, k)

        Bx_H0, Bx_V0, Bx_nHu, Bx_nVu, Bx_has, Bx_cl = plane_placement(
            Bx_offs, crops, Bx_shape)
        Jz_H0, Jz_V0, Jz_nHu, Jz_nVu, Jz_has, Jz_cl = plane_placement(
            Jz_offs, crops, Jz_shape)

        if not (Bx_has or Jz_has):
            continue

        with h5py.File(fp, "r") as f:
            if Bx_has:
                path = f"fields/Bx/{cycle_name}"
                if path not in f:
                    raise KeyError(f"Missing dataset: {path}")
                Bx_slab = read_field_plane(f, path, crops[0], crops[1], crops[2],
                                           plane, Bx_cl)
                local_Bx[Bx_H0:Bx_H0 + Bx_nHu, Bx_V0:Bx_V0 + Bx_nVu] = Bx_slab
                local_Bx_occ[Bx_H0:Bx_H0 + Bx_nHu, Bx_V0:Bx_V0 + Bx_nVu] += 1

            if Jz_has:
                Jz_slab = read_current_plane(f, cycle_name, species_list,
                                             crops[0], crops[1], crops[2], plane,
                                             Jz_cl, current_mode)
                local_Jz[Jz_H0:Jz_H0 + Jz_nHu, Jz_V0:Jz_V0 + Jz_nVu] = Jz_slab
                local_Jz_occ[Jz_H0:Jz_H0 + Jz_nHu, Jz_V0:Jz_V0 + Jz_nVu] += 1

    Bx = None
    Jz = None
    Bx_occ = None
    Jz_occ = None

    if rank == 0:
        Bx = np.zeros((Bx_H, Bx_V), dtype=np.float64)
        Jz = np.zeros((Jz_H, Jz_V), dtype=np.float64)
        Bx_occ = np.zeros((Bx_H, Bx_V), dtype=np.int32)
        Jz_occ = np.zeros((Jz_H, Jz_V), dtype=np.int32)

    comm.Reduce(local_Bx, Bx, op=MPI.SUM, root=0)
    comm.Reduce(local_Jz, Jz, op=MPI.SUM, root=0)
    comm.Reduce(local_Bx_occ, Bx_occ, op=MPI.SUM, root=0)
    comm.Reduce(local_Jz_occ, Jz_occ, op=MPI.SUM, root=0)

    return Bx, Jz, Bx_occ, Jz_occ


def plot_cycle_2d(cycle_name, Bx, Jz, Bx_occ, Jz_occ, outdir, plane, extents):
    ###! Panel axes depend on the resolved plane:
    ###!   XY -> H=x, V=y   extent [xmin,xmax, ymin,ymax]
    ###!   YZ -> H=z, V=y   extent [zmin,zmax, ymin,ymax]
    ###!   ZX -> H=z, V=x   extent [zmin,zmax, xmin,xmax]
    xmin, xmax, ymin, ymax, zmin, zmax = extents

    if Bx_occ.min() == 0:
        print(f"WARNING: {cycle_name} Bx has gaps.", flush=True)
    if Jz_occ.min() == 0:
        print(f"WARNING: {cycle_name} Jz has gaps.", flush=True)

    if plane == "XY":
        h_ext = (xmin, xmax); v_ext = (ymin, ymax)
        h_lab = r"$x\,\omega_p/c$"; v_lab = r"$y\,\omega_p/c$"
        bx_ttl = r"$B_x$ $(X,Y)$"; jz_ttl = r"$J_z$ $(X,Y)$"
    elif plane == "YZ":
        h_ext = (zmin, zmax); v_ext = (ymin, ymax)
        h_lab = r"$z\,\omega_p/c$"; v_lab = r"$y\,\omega_p/c$"
        bx_ttl = r"$B_x$ $(Z,Y)$"; jz_ttl = r"$J_z$ $(Z,Y)$"
    elif plane == "ZX":
        h_ext = (zmin, zmax); v_ext = (xmin, xmax)
        h_lab = r"$z\,\omega_p/c$"; v_lab = r"$x\,\omega_p/c$"
        bx_ttl = r"$B_x$ $(Z,X)$"; jz_ttl = r"$J_z$ $(Z,X)$"
    else:
        raise ValueError(f"Unknown plane '{plane}'")

    extent = [h_ext[0], h_ext[1], v_ext[0], v_ext[1]]

    fig, axs = plt.subplots(1, 2, figsize=(11, 8), dpi=200)

    ###! imshow arr.T so arr's first axis (H) is horizontal, second (V) vertical.
    im0 = axs[0].imshow(Bx.T, origin="lower", cmap="seismic", aspect="auto",
                        vmin=-0.25, vmax=0.25, extent=extent)
    axs[0].set_title(bx_ttl, fontsize=16)
    axs[0].set_xlabel(h_lab, fontsize=12)
    axs[0].set_ylabel(v_lab, fontsize=12)
    axs[0].tick_params(axis="both", which="major", labelsize=12, length=6)
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(Jz.T, origin="lower", cmap="seismic", aspect="auto",
                        vmin=-0.08, vmax=0.08, extent=extent)
    axs[1].set_title(jz_ttl, fontsize=16)
    axs[1].set_xlabel(h_lab, fontsize=12)
    axs[1].set_ylabel(v_lab, fontsize=12)
    axs[1].tick_params(axis="both", which="major", labelsize=12, length=6)
    fig.colorbar(im1, ax=axs[1])

    cycle_number = int(cycle_name.replace("cycle_", ""))
    time_omega = cycle_number / args.cycle_step * args.time_step
    fig.suptitle(rf"$T = {time_omega}\,\omega_p^{{-1}}$", fontsize=18)
    fig.tight_layout()

    outfile = os.path.join(outdir, f"{cycle_name}.png")
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {outfile}", flush=True)


###! ============================================================
###! Arguments
###! ============================================================

parser = argparse.ArgumentParser(
    description="MPI plotter for Bx and Jz from iPIC3D output. 3D (default): "
                "3 panels Bx(XY), Jz(XY) at the z-midplane and Jz(ZY) at the "
                "x-midplane. 2D-spatial (--plane XY|YZ|ZX): 2 panels Bx, Jz on "
                "the resolved plane. Decomposition derived from cell counts + "
                "tile shape; Jz layout (total vs per-species) auto-detected.")

parser.add_argument("dir_data", type=str)

###! Physical box extents (positional, same order/style as before)
parser.add_argument("xmin", type=float)
parser.add_argument("xmax", type=float)
parser.add_argument("ymin", type=float)
parser.add_argument("ymax", type=float)
parser.add_argument("zmin", type=float)
parser.add_argument("zmax", type=float)

###! Cell counts (required, both modes). A collapsed 2D axis has nc = 1.
parser.add_argument("--nxc", type=int, required=True, help="Number of cells in x")
parser.add_argument("--nyc", type=int, required=True, help="Number of cells in y")
parser.add_argument("--nzc", type=int, required=True, help="Number of cells in z")

###! Mode selector. Omitted -> 3D (3 panels). XY|YZ|ZX -> 2D-spatial (2 panels),
###! collapsing Z|X|Y respectively; that axis must have nc = 1.
parser.add_argument("--plane", type=str, default=None, choices=["XY", "YZ", "ZX"],
                    help="Omit for 3D (original 3-panel behaviour). Set to select "
                         "a 2D-spatial run's resolved plane.")

parser.add_argument("--cycle-start", type=int, default=0)
parser.add_argument("--cycle-end", type=int, default=20000)
parser.add_argument("--cycle-step", type=int, default=500)
parser.add_argument("--time-step", type=float, default=50.0)

parser.add_argument("--species", type=int, nargs="+", default=[1, 2])

parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--mapping", type=str, default="auto",
                    choices=["auto", "A", "B", "C", "D", "E", "F"])

args = parser.parse_args()

dir_data = args.dir_data
nxc, nyc, nzc = args.nxc, args.nyc, args.nzc
plane = args.plane
species_list = args.species
extents = (args.xmin, args.xmax, args.ymin, args.ymax, args.zmin, args.zmax)

###! ---- resolve and validate the mode (both conditions must agree) ----
###! 2D requires --plane AND the collapsed axis nc == 1.
###! 3D requires no --plane AND all nc > 1.
coll_axis_of = {"XY": 2, "YZ": 0, "ZX": 1}
ncs = (nxc, nyc, nzc)
any_collapsed = any(n == 1 for n in ncs)

if plane is not None:
    ###! 2D requested: the named plane's collapsed axis must be the nc == 1 one.
    ca = coll_axis_of[plane]
    if ncs[ca] != 1:
        raise SystemExit(
            f"--plane {plane} collapses {'xyz'[ca]}, but n{'xyz'[ca]}c = {ncs[ca]} "
            f"(expected 1 for a 2D-spatial run). Mode conditions disagree.")
    other = [a for a in range(3) if a != ca]
    for a in other:
        if ncs[a] == 1:
            raise SystemExit(
                f"--plane {plane} but n{'xyz'[a]}c = 1 as well: more than one axis "
                f"is collapsed, which is not a 2D-spatial plane.")
    mode = "2d"
else:
    ###! 3D requested: no axis may be collapsed.
    if any_collapsed:
        bad = [f"n{'xyz'[a]}c=1" for a in range(3) if ncs[a] == 1]
        raise SystemExit(
            f"No --plane given (3D mode) but {', '.join(bad)} indicates a collapsed "
            f"axis. For a 2D-spatial run pass --plane "
            f"{'/'.join(k for k, v in coll_axis_of.items())}. Mode conditions disagree.")
    mode = "3d"

if args.outdir is None:
    outdir = dir_data
else:
    outdir = args.outdir

if rank == 0:
    os.makedirs(outdir, exist_ok=True)

first_cycle = f"cycle_{args.cycle_start}"

###! ============================================================
###! rank 0: discover files, probe tile shape, DERIVE decomposition, map, layout
###! ============================================================
if rank == 0:
    all_files = sorted(glob.glob(os.path.join(dir_data, "proc*.hdf")))

    if len(all_files) == 0:
        raise RuntimeError(f"No proc*.hdf files found in {dir_data}")

    with h5py.File(all_files[0], "r") as f:
        Bx_path = f"fields/Bx/{first_cycle}"
        if Bx_path not in f:
            raise KeyError(f"Missing dataset in sample file: {Bx_path}")
        Bx_shape = f[Bx_path].shape

        ###! Detect current layout: total (moments/Jz) vs per-species
        total_path = f"moments/Jz/{first_cycle}"
        species_path = f"moments/species_{species_list[0]}/Jz/{first_cycle}"

        if total_path in f:
            current_mode = "total"
            Jz_shape = f[total_path].shape
        elif species_path in f:
            current_mode = "species"
            Jz_shape = f[species_path].shape
        else:
            raise KeyError(
                f"Found neither '{total_path}' nor '{species_path}' in sample "
                f"file. Cannot locate Jz.")

    ###! ---- derive the MPI decomposition from cell counts + tile shape ----
    ###! LEN = nc / (n_tile - 1). A collapsed axis has nc = 1, n_tile = 2 -> 1.
    nx_t, ny_t, nz_t = Bx_shape
    for lbl, n_c, n_t in (("x", nxc, nx_t), ("y", nyc, ny_t), ("z", nzc, nz_t)):
        if n_t < 2 or n_c % (n_t - 1) != 0:
            raise RuntimeError(
                f"Cannot derive the {lbl} decomposition: {n_c} cells do not "
                f"divide evenly into tiles of {n_t - 1} cells (tile shape "
                f"{Bx_shape}). Check --n{lbl}c or the shared-boundary assumption.")

    XLEN = nxc // (nx_t - 1)
    YLEN = nyc // (ny_t - 1)
    ZLEN = nzc // (nz_t - 1)

    if XLEN * YLEN * ZLEN != len(all_files):
        raise RuntimeError(
            f"Derived decomposition {XLEN}x{YLEN}x{ZLEN} = {XLEN*YLEN*ZLEN} "
            f"tiles, but {len(all_files)} proc*.hdf files are present. Cell "
            f"counts, tile shape and file count disagree -- stopping rather "
            f"than assembling a corrupt field.")

    ###! moment tile shape must match the field tile shape or the shared-plane
    ###! assembly indexing (reused for Jz) would be wrong.
    with h5py.File(all_files[0], "r") as f:
        jz_first = jz_path(current_mode, species_list[0], first_cycle)
        ms = tuple(f[jz_first].shape)
    if ms != tuple(Bx_shape):
        raise RuntimeError(
            f"Jz tile shape {ms} differs from Bx tile shape {tuple(Bx_shape)}; "
            f"the shared assembly indexing would be wrong.")

    Bx_global_shape = global_shape_shared(Bx_shape, XLEN, YLEN, ZLEN)
    Jz_global_shape = global_shape_shared(Jz_shape, XLEN, YLEN, ZLEN)

    if args.mapping == "auto":
        map_name, map_score = choose_mapping(all_files, XLEN, YLEN, ZLEN)
    else:
        map_name = args.mapping
        map_score = -1

    print(f"Mode            : {mode.upper()}"
          f"{'  plane=' + plane if mode == '2d' else '  (3 panels)'}", flush=True)
    print(f"Decomposition   : {XLEN} x {YLEN} x {ZLEN} tiles (derived) "
          f"= {XLEN*YLEN*ZLEN} files", flush=True)
    print(f"Tile shape      : {tuple(Bx_shape)}", flush=True)
    print(f"Current layout  : "
          f"{'total (single dataset; --species ignored)' if current_mode == 'total' else 'species (summing ' + str(species_list) + ')'}",
          flush=True)
    print(f"Mapping         : {map_name}  (score {map_score}; 0 is perfect)",
          flush=True)

else:
    all_files = None
    Bx_shape = None
    Jz_shape = None
    Bx_global_shape = None
    Jz_global_shape = None
    current_mode = None
    map_name = None
    XLEN = YLEN = ZLEN = None

all_files = comm.bcast(all_files, root=0)
Bx_shape = comm.bcast(Bx_shape, root=0)
Jz_shape = comm.bcast(Jz_shape, root=0)
Bx_global_shape = comm.bcast(Bx_global_shape, root=0)
Jz_global_shape = comm.bcast(Jz_global_shape, root=0)
current_mode = comm.bcast(current_mode, root=0)
map_name = comm.bcast(map_name, root=0)
XLEN = comm.bcast(XLEN, root=0)
YLEN = comm.bcast(YLEN, root=0)
ZLEN = comm.bcast(ZLEN, root=0)

maps = mapping_candidates(XLEN, YLEN, ZLEN)
rank_to_ijk = maps[map_name]

local_files = all_files[rank::size]

Bx_nx, Bx_ny, Bx_nz = Bx_global_shape
Jz_nx, Jz_ny, Jz_nz = Jz_global_shape

###! ---- mode-specific slice indices ----
if mode == "3d":
    ###! ALWAYS slice at the global midplane (original behaviour):
    ###!   XY plane at z = nz_global // 2 ; ZY plane at x = nx_global // 2
    z_index = Bx_nz // 2
    x_index = Jz_nx // 2

    if not (0 <= z_index < Bx_nz):
        raise ValueError(f"midplane z_index={z_index} outside Bx z range "
                         f"[0,{Bx_nz - 1}]")
    if not (0 <= z_index < Jz_nz):
        raise ValueError(f"midplane z_index={z_index} outside Jz z range "
                         f"[0,{Jz_nz - 1}] (Bx and Jz have different global z)")
    if not (0 <= x_index < Jz_nx):
        raise ValueError(f"midplane x_index={x_index} outside Jz x range "
                         f"[0,{Jz_nx - 1}]")

    if rank == 0:
        print(f"Midplane slices : XY at z={z_index} (of {Bx_nz}), "
              f"ZY at x={x_index} (of {Jz_nx})", flush=True)
else:
    ###! 2D: sample the lower global node of the collapsed axis (2-node pair).
    coll_index = 0
    if rank == 0:
        ca = coll_axis_of[plane]
        print(f"Global nodes    : Bx {Bx_nx}x{Bx_ny}x{Bx_nz}  "
              f"Jz {Jz_nx}x{Jz_ny}x{Jz_nz}", flush=True)
        print(f"Collapsed node  : {'xyz'[ca]} index {coll_index}", flush=True)

cycles = list(range(args.cycle_start, args.cycle_end + 1, args.cycle_step))

for cycle in cycles:
    cycle_name = f"cycle_{cycle}"

    if rank == 0:
        print("", flush=True)
        print(f"Processing {cycle_name}", flush=True)

    if mode == "3d":
        Bx_XY, Jz_XY, Jz_ZY, Bx_occ, Jz_XY_occ, Jz_ZY_occ = assemble_cycle_3d(
            cycle_name, local_files, rank_to_ijk, Bx_shape, Jz_shape,
            Bx_global_shape, Jz_global_shape, z_index, x_index,
            species_list, current_mode)

        if rank == 0:
            plot_cycle_3d(cycle_name, Bx_XY, Jz_XY, Jz_ZY,
                          Bx_occ, Jz_XY_occ, Jz_ZY_occ, outdir, extents)
    else:
        Bx, Jz, Bx_occ, Jz_occ = assemble_cycle_2d(
            cycle_name, local_files, rank_to_ijk, Bx_shape, Jz_shape,
            Bx_global_shape, Jz_global_shape, plane, coll_index,
            species_list, current_mode)

        if rank == 0:
            plot_cycle_2d(cycle_name, Bx, Jz, Bx_occ, Jz_occ, outdir, plane,
                          extents)

if rank == 0:
    print("", flush=True)
    print(f"All plots saved in: {outdir}", flush=True)
    print(f"Complete. Time elapsed = {datetime.now() - start_time}", flush=True)