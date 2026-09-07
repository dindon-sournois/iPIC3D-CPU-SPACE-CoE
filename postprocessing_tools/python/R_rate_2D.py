"""
Created on Sun Sep 07 2026

@author: Pranab JD, Claude

Description: Reconnection rate / flow diagnostic for a 2D double-Harris iPIC3D
             run, in EITHER the XY plane OR the YZ plane. This is the 2D
             counterpart of Reconnection_rate_3D.py; it copies that script's
             validated flux-function machinery but reads a SINGLE resolved plane
             from a run whose third axis is collapsed (one tile, sampled at its
             lower global node), following the assembly pattern of
             Spectra_compute.py exactly.

             Geometry convention (fixed by the runs, do NOT re-derive):

                 inflow  is ALWAYS along y            -> v_in from Vy
                 the sheet lies along the RESOLVED horizontal axis H
                 the normal is y

                 XY plane : H = x, collapse z, outflow = Vx, sheet reverses in Bx
                 YZ plane : H = z, collapse x, outflow = Vz, sheet reverses in Bz

             *** RECONNECTION EXISTS ONLY IN THE XY PLANE. ***

             XY (--plane xy):
                 Full reconnection measurement, identical in spirit to the 3D
                 script on a single plane:
                   * flux function psi(x,y) from (Bx, By), Bx = +d_y psi,
                     By = -d_x psi, solved spectrally (fully periodic in-plane).
                   * neutral-line X/O identification, persistent-ID tracking,
                     per-pair signed rate.
                   * flux-based rate R = (1/(B0 vA)) d(Delta psi)/dt and the
                     independent Ez-based rate.
                   * paper-method rate v_rec = <v_in>/v_out (Appendix G) with
                     inflow = Vy, outflow = Vx, from the mass-weighted
                     single-fluid velocity.

             YZ (--plane yz):
                 There is NO reconnection in this plane, so NO flux function, NO
                 X/O points, NO psi-rate and NO Ez-rate are computed. ONLY the
                 paper-method inflow/outflow FLOW diagnostic is produced, with
                 inflow = Vy and outflow = Vz. It is written to a file whose name
                 and header state in plain terms that it is a FLOW DIAGNOSTIC
                 (drift-kink / outflow characterisation) and is NOT a
                 reconnection rate. The ratio <v_in>/v_out is reported for
                 completeness but must not be read as a reconnection rate.

Single-plane assembly (from Spectra_compute.py):
  The collapsed axis has LEN = 1 (a single tile). Global nodes on it number
  n_tile - 1 + 1; we sample coll_index = 0, the physical plane (node 1, if it
  exists, is its periodic image). NO averaging over the collapsed axis is done
  or implied, so every z-average / per-plane-coherence / --z-planes feature of
  the 3D script is REMOVED here: in 2D there is exactly one plane and the
  measure-then-average vs average-then-measure distinction is empty (coherence
  == 1 identically).

Velocity (paper method, both planes use Vy for inflow):
  Mass-weighted single-fluid velocity from ALL four species,

      V = sum_s (m_s/q_s) J_s  /  sum_s (m_s/q_s) rho_s

  with m_s/q_s = -1 for electron species (0, 2) and +mass_ratio for proton
  species (1, 3); the common 1/e cancels. XY needs (Jy, Jx); YZ needs (Jy, Jz).
  Only the components the plane actually uses are read.

Species layout (fixed by the run):
    0 = background electrons    1 = background protons
    2 = current-sheet electrons 3 = current-sheet protons

Usage
-----
  SCRIPT="../postprocessing_tools/python/Reconnection_rate_2D.py"
  DATA_DIR="/scratch/.../sigma_5/"
  OUT_DIR="${DATA_DIR}/reconnection"

  # 2D XY run: nzc = 1 (z collapsed).  Extents still span the full box.
  xmin=0; xmax=512
  ymin=0; ymax=1024
  zmin=0; zmax=0.1

  srun python3 "$SCRIPT" "$DATA_DIR" "$OUT_DIR" \
      $xmin $xmax $ymin $ymax $zmin $zmax \
      --plane xy \
      --nxc 4096 --nyc 8192 --nzc 1 \
      --sigma 5 --time-denom 10 --mapping A \
      --mass-ratio 1836 \
      --cycle-start 0 --cycle-end 20000 --cycle-step 500 --cycle-chunk 2 \
      --ez-smooth 21

  # 2D YZ run: nxc = 1 (x collapsed).  FLOW DIAGNOSTIC ONLY.
  srun python3 "$SCRIPT" "$DATA_DIR" "$OUT_DIR" \
      0 0.1  0 1024  0 512 \
      --plane yz \
      --nxc 1 --nyc 8192 --nzc 4096 \
      --sigma 5 --time-denom 10 --mapping A --mass-ratio 1836 \
      --cycle-start 0 --cycle-end 20000 --cycle-step 500 --cycle-chunk 2

Outputs
-------
  XY (--plane xy):
     R_rate_dAz_dt.txt/.png         flux-based rate from (Bx,By): d(Delta psi)/dt
                                    and the Ez-based rate. (Named for continuity
                                    with the 3D script; in 2D there is only the
                                    one plane, so "field_avg" is just "the plane".)
     R_rate_vout_vin.txt/.png                     paper-method v_rec = <v_in>/v_out, inflow=Vy,
                                    outflow=Vx. Written iff per-species moments
                                    are present.
     reconnection_rate_xo_pairs.txt per-X/O-pair signed rate.
     xo_check.txt                   (with --check-xo) X/O verification.

  YZ (--plane yz):
     flow_diagnostic_YZ.txt/.png    inflow=Vy, outflow=Vz FLOW diagnostic. NOT a
                                    reconnection rate -- the header says so.
                                    Written iff per-species moments are present.
                                    NOTHING flux-based is written for YZ.
"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import glob
import argparse
from datetime import datetime

import numpy as np
import h5py
from mpi4py import MPI

###! Plotting is rank-0 only and headless.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

t_wall = datetime.now()


###! ============================================================
###! Paper-method configuration (Appendix G box fractions)
###! ============================================================
###! Scaled from the paper's L-normalised boxes to THIS domain, EXACTLY as in
###! the 3D script. The sheet-to-boundary half-width per sheet is Ly/4 (two
###! sheets, one per y-half). These are plane-independent: inflow is always y.
INFLOW_FRAC_LO   = 0.30      ###! inner edge of the inflow band (y), in units of Ly/4
INFLOW_FRAC_HI   = 0.80      ###! outer edge of the inflow band (y), in units of Ly/4
INFLOW_H_LO      = 0.20      ###! along-sheet (H) inner edge of the inflow box, in LH
INFLOW_H_HI      = 0.80      ###! along-sheet (H) outer edge of the inflow box, in LH
                             ###! Makes the inflow region a true RECTANGLE: bounded
                             ###! in y (0.3..0.8 Ly/4 from the sheet, both sides)
                             ###! AND in the along-sheet coord H (0.2..0.8 LH). H is
                             ###! x for XY, z for YZ. Set 0.0/1.0 for full H.
OUTFLOW_HALF_FRAC = 0.25     ###! outflow band half-height (y), in units of Ly/4
OUTFLOW_H_GUARD   = 0.10     ###! drop this fraction of LH from EACH H-end before
                             ###! taking max|v_out|. H is PERIODIC (x or z), so this
                             ###! removes no boundary artifact; it only shrinks the
                             ###! max window and can bias v_rec UP if a jet sits near
                             ###! H=0 or H=LH. 0.0 recovers the full H-range.

###! Species indices (fixed by the run). Electrons carry negative charge.
SPECIES_ELECTRON = (0, 2)    ###! background + current-sheet electrons
SPECIES_PROTON   = (1, 3)    ###! background + current-sheet protons
ALL_SPECIES      = (0, 1, 2, 3)

###! Which physical axis each plane COLLAPSES (matches Spectra_compute.py's
###! COLL_AXIS_OF). XY collapses z (index 2); YZ collapses x (index 0). y (index
###! 1) is resolved in both.
COLL_AXIS_OF = {"xy": 2, "yz": 0}


###! ============================================================
###! Tile / mapping helpers  (identical to the 3D and Spectra scripts)
###! ============================================================

def proc_id_from_filename(fp):
    base = os.path.basename(fp)
    return int(base.replace("proc", "").replace(".hdf", ""))


def mapping_candidates(XLEN, YLEN, ZLEN):
    """proc_id -> (i,j,k). Six common orderings; the right one is inferred."""
    def A(pid):
        k = pid % ZLEN; t = pid // ZLEN
        j = t % YLEN;   i = t // YLEN
        return i, j, k

    def B(pid):
        j = pid % YLEN; t = pid // YLEN
        k = t % ZLEN;   i = t // ZLEN
        return i, j, k

    def C(pid):
        k = pid % ZLEN; t = pid // ZLEN
        i = t % XLEN;   j = t // XLEN
        return i, j, k

    def D(pid):
        i = pid % XLEN; t = pid // XLEN
        j = t % YLEN;   k = t // YLEN
        return i, j, k

    def E(pid):
        j = pid % YLEN; t = pid // YLEN
        i = t % XLEN;   k = t // XLEN
        return i, j, k

    def F(pid):
        i = pid % XLEN; t = pid // XLEN
        k = t % ZLEN;   j = t // ZLEN
        return i, j, k

    return {"A": A, "B": B, "C": C, "D": D, "E": E, "F": F}


def choose_mapping(files, XLEN, YLEN, ZLEN):
    """
    Score each candidate proc -> (i,j,k) by grid occupancy; the correct one
    covers every tile exactly once (score 0).

    NOTE (copied from Spectra_compute.py): with a collapsed axis (LEN == 1)
    several candidates tie, because permuting a length-1 axis changes nothing.
    On a tie this returns the first best; pass --mapping to force one if the auto
    choice is ever wrong for your file ordering. Unlike the 3D script we do NOT
    fall through to a field-smoothness tiebreak: for a single plane it is cheap
    for the user to pass --mapping, and the smoothness tiebreak assumed three
    resolved axes.
    """
    proc_ids = [proc_id_from_filename(fp) for fp in files]
    maps = mapping_candidates(XLEN, YLEN, ZLEN)
    best_name, best_score = None, None
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
        score = 10 * int(np.count_nonzero(occ == 0)) + \
                100 * int(np.count_nonzero(occ > 1))
        if best_score is None or score < best_score:
            best_score, best_name = score, name
    if best_name is None:
        raise RuntimeError("Could not determine proc -> (i,j,k) mapping.")
    return best_name


###! ============================================================
###! Single-plane assembly  (pattern copied from Spectra_compute.py)
###! ============================================================
###! One resolved plane (H, y): the collapsed axis is sampled at its single
###! global node coll_index = 0, and the along-sheet axis H is returned FIRST so
###! the downstream code sees the same (along-sheet, normal) = (axis0, axis1)
###! layout in both planes -- exactly as the 3D script had (x, y).
###!
###!   plane xy: H = x (collapse z) -> arrays (Gx, Gy)      read [xs:, ys:, 0]
###!   plane yz: H = z (collapse x) -> arrays (Gz, Gy)      read [0, ys:, zs:].T
###!
###! Duplicate shared-boundary planes are averaged via a per-node count, so the
###! HDF5 is opened ONCE per tile. Returns rank-0 dict of (GH, Gy) arrays.
###! ============================================================

def assemble_plane_chunk(cycles, local_files, rank_to_ijk, tile_shape,
                         G_shape, datasets, plane, coll_index, work_dtype):
    """
    COLLECTIVE. Assemble every dataset in `datasets` on the resolved plane, for
    a CHUNK of cycles, into (nc, GH, Gy) arrays on rank 0.

    `datasets` maps key -> HDF5 path template containing '{cycle}'. A key whose
    template is MISSING for a given cycle is reported (added to a per-rank miss
    list) and that cycle is skipped for that key; a key missing on a tile that
    should own it aborts collectively (a hole would be silently averaged over).

    Returns on rank 0: dict(fields={key:(nc,GH,Gy)}, found=(nc,) int, ntiles).
    None on other ranks. `found[ci]` counts tiles that supplied cycle ci for the
    FIRST dataset key (the reference field); it is set to -1 for a cycle present
    in some-but-not-all tiles (incomplete -> invalid, matches the 3D policy).
    """
    Gx, Gy, Gz = G_shape
    nx_t, ny_t, nz_t = tile_shape
    nx_c, ny_c, nz_c = nx_t - 1, ny_t - 1, nz_t - 1
    nc = len(cycles)

    ###! resolved horizontal (H) global size and the raw cell size along it
    if plane == "xy":
        GH = Gx
    else:                                    ###! yz
        GH = Gz

    keys = list(datasets.keys())
    ref_key = keys[0]                         ###! defines "present" for a cycle

    loc = {k: np.zeros((nc, GH, Gy), dtype=work_dtype) for k in keys}
    loccnt = np.zeros((GH, Gy), dtype=np.int32)      ###! collapsed-nodes per (H,y) col
    locfound = np.zeros(nc, dtype=np.int32)
    locfail = np.zeros(1, dtype=np.int32)
    loctiles = np.zeros(1, dtype=np.int32)
    missing_path = [None]

    for fp in local_files:
        i, j, k = rank_to_ijk(proc_id_from_filename(fp))
        xs = 0 if i == 0 else 1
        ys = 0 if j == 0 else 1
        zs = 0 if k == 0 else 1
        gx0, gy0, gz0 = i * nx_c + xs, j * ny_c + ys, k * nz_c + zs
        nxu, nyu, nzu = nx_t - xs, ny_t - ys, nz_t - zs

        ###! ---- collapsed-axis ownership: does this tile hold coll_index? ----
        ###! coll_index is a GLOBAL node on the collapsed axis (always 0). The
        ###! local index is coll_index - (uncropped tile origin). Same logic as
        ###! Spectra_compute.py.
        if plane == "xy":                     ###! collapse z
            c_o = k * nz_c                    ###! uncropped z origin of this tile
            c_lo = gz0                        ###! cropped lower owned node
            c_hi = k * nz_c + nz_t - 1        ###! uncropped upper owned node
            h_g0, h_nu = gx0, nxu             ###! H = x, output-first axis
        else:                                 ###! yz, collapse x
            c_o = i * nx_c
            c_lo = gx0
            c_hi = i * nx_c + nx_t - 1
            h_g0, h_nu = gz0, nzu             ###! H = z, output-first axis

        if not (c_lo <= coll_index <= c_hi):
            continue
        c_local = coll_index - c_o            ###! local index on the collapsed axis

        loctiles[0] += 1

        ###! NEVER let an I/O error escape before the collective Allreduce below.
        try:
            with h5py.File(fp, "r") as f:
                for ci, cyc in enumerate(cycles):
                    ###! presence defined by the reference field of this cycle
                    ref_path = datasets[ref_key].format(cycle=cyc)
                    if ref_path not in f:
                        continue
                    for key, tmpl in datasets.items():
                        path = tmpl.format(cycle=cyc)
                        if path not in f:
                            missing_path[0] = f"{os.path.basename(fp)}:{path}"
                            raise KeyError(missing_path[0])
                        d = f[path]
                        if plane == "xy":
                            ###! keep (x, y) at fixed z = c_local
                            blk = np.asarray(d[xs:, ys:, c_local], dtype=np.float64)   ###! [x, y]
                        else:
                            ###! keep (z, y) at fixed x = c_local; read [y, z] -> T
                            slab = np.asarray(d[c_local, ys:, zs:], dtype=np.float64)   ###! [y, z]
                            blk = slab.T                                               ###! [z, y]
                        loc[key][ci, h_g0:h_g0 + h_nu, gy0:gy0 + nyu] += blk
                    locfound[ci] += 1
        except KeyError:
            locfail[0] += 1
            break
        except Exception as e:
            locfail[0] += 1
            missing_path[0] = f"{os.path.basename(fp)}: read error: {e}"
            break

        loccnt[h_g0:h_g0 + h_nu, gy0:gy0 + nyu] += 1

    ###! ---- collective abort on ANY missing dataset / read error ----
    nfail = comm.allreduce(int(locfail[0]), op=MPI.SUM)
    if nfail > 0:
        all_missing = comm.gather(missing_path[0], root=0)
        fatal = None
        if rank == 0:
            hits = [m for m in all_missing if m is not None]
            first = hits[0] if hits else "unknown dataset"
            fatal = (f"Plane assembly failed: {first}. Every requested dataset "
                     f"must exist for every requested cycle on every tile that "
                     f"owns the plane. Aborting rather than averaging over a hole.")
        fatal = comm.bcast(fatal, root=0)
        raise RuntimeError(fatal)

    ###! ---- reduce to root ----
    mpi_t = MPI.FLOAT if work_dtype == np.float32 else MPI.DOUBLE

    def red_f(arr):
        out = np.zeros_like(arr) if rank == 0 else None
        comm.Reduce([arr, mpi_t], [out, mpi_t] if rank == 0 else None,
                    op=MPI.SUM, root=0)
        return out

    fields = {k: red_f(loc[k]) for k in keys}

    cnt = np.zeros_like(loccnt) if rank == 0 else None
    comm.Reduce([loccnt, MPI.INT], [cnt, MPI.INT] if rank == 0 else None,
                op=MPI.SUM, root=0)
    found = np.zeros_like(locfound) if rank == 0 else None
    comm.Reduce([locfound, MPI.INT], [found, MPI.INT] if rank == 0 else None,
                op=MPI.SUM, root=0)
    ntiles = comm.allreduce(int(loctiles[0]), op=MPI.SUM)

    if rank != 0:
        return None

    if cnt.min() == 0:
        raise RuntimeError("Plane assembly gap: some (H,y) nodes received no "
                           "data. Check the mapping or the collapsed-axis index.")

    ###! Duplicate shared boundaries: divide each field by the per-column count.
    ###! On the collapsed axis the count is 1 everywhere (single plane); on the
    ###! resolved seams it is >1 where tiles share a boundary node, exactly as in
    ###! the Spectra assembler. Divide the SUM by the count to recover the value.
    for k in keys:
        fields[k] /= cnt[None, :, :]

    ###! cycles present in some-but-not-all owning tiles -> invalid
    for ci in range(nc):
        if 0 < found[ci] < ntiles:
            found[ci] = -1

    return dict(fields=fields, found=found, ntiles=ntiles)


###! ============================================================
###! Mass-weighted single-fluid velocity from the assembled moments
###! ============================================================

def build_velocity(fields, ci, mass_ratio, comp_out):
    """
    Form the z/x-collapsed mass-weighted bulk velocity on the plane, for cycle
    index ci, from the per-species rho and current components already assembled
    into `fields`.

        V_a = sum_s (m_s/q_s) J_{a,s}  /  sum_s (m_s/q_s) rho_s

    with weight -1 for electrons (0,2), +mass_ratio for protons (1,3). Returns
    (V_normal, V_out) = (Vy, V_H) where V_H is Vx for XY and Vz for YZ.

    `fields` must contain keys 'rho_s{S}', 'Jy_s{S}', and '{comp_out}_s{S}' for
    every species S. comp_out is 'Jx' (XY) or 'Jz' (YZ).
    """
    def weight(s):
        if s in SPECIES_ELECTRON:
            return -1.0
        if s in SPECIES_PROTON:
            return float(mass_ratio)
        raise RuntimeError(f"Species {s} is neither electron nor proton.")

    MFy = None; MFo = None; Mrho = None
    for s in ALL_SPECIES:
        w = weight(s)
        rho = fields[f"rho_s{s}"][ci]
        jy  = fields[f"Jy_s{s}"][ci]
        jo  = fields[f"{comp_out}_s{s}"][ci]
        Mrho = w * rho if Mrho is None else Mrho + w * rho
        MFy  = w * jy  if MFy  is None else MFy  + w * jy
        MFo  = w * jo  if MFo  is None else MFo  + w * jo

    rho_floor = 1e-30
    safe = np.abs(Mrho) > rho_floor
    Vy = np.where(safe, MFy / Mrho, 0.0)
    Vo = np.where(safe, MFo / Mrho, 0.0)
    return Vy, Vo


###! ============================================================
###! Flux function and neutral-line extraction  (XY only)
###! ============================================================
###! These are copied verbatim from Reconnection_rate_3D.py with the along-sheet
###! axis renamed x->H internally where it matters, but for XY H IS x, so the
###! bodies are unchanged. They are called ONLY on the XY plane.

def _cumtrapz0(y, d, axis):
    """Cumulative trapezoid with a leading zero (avoids a scipy dependency)."""
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[axis]
    lo = np.take(y, np.arange(0, n - 1), axis=axis)
    hi = np.take(y, np.arange(1, n), axis=axis)
    c = np.cumsum(0.5 * d * (lo + hi), axis=axis)
    pad = list(y.shape)
    pad[axis] = 1
    return np.concatenate([np.zeros(pad), c], axis=axis)


def flux_function_fft(Bx, By, dx, dy, nxc, nyc):
    """
    Flux function by spectral solution of the Poisson equation, for a FULLY
    PERIODIC in-plane field. From Bx = +d_y psi, By = -d_x psi:

        d_y Bx - d_x By = laplacian(psi)

    psi is single-valued by construction and the compressive (curl-free) part of
    B is projected out. Valid for a single 2D periodic plane, which is exactly
    what an XY 2D run provides. Returns psi on the (nxc+1, nyc+1) node grid.
    """
    bx = Bx[:nxc, :nyc]
    by = By[:nxc, :nyc]
    kx = 2.0 * np.pi * np.fft.fftfreq(nxc, d=dx)[:, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(nyc, d=dy)[None, :]
    omega_k = 1j * ky * np.fft.fft2(bx) - 1j * kx * np.fft.fft2(by)
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0
    psi_k = -omega_k / k2
    psi_k[0, 0] = 0.0
    core = np.real(np.fft.ifft2(psi_k))
    psi = np.empty((nxc + 1, nyc + 1), dtype=np.float64)
    psi[:nxc, :nyc] = core
    psi[nxc, :nyc] = core[0, :]
    psi[:nxc, nyc] = core[:, 0]
    psi[nxc, nyc] = core[0, 0]
    return psi


def bx_from_psi(psi, dy, nxc, nyc):
    """The Bx implied by psi, Bx_sol = d_y psi, spectral on the periodic grid.
    The neutral line MUST be found from this (the solenoidal projection), not the
    raw Bx, or a compressive part injects spurious Bx=0 crossings."""
    ky = 2.0 * np.pi * np.fft.fftfreq(nyc, d=dy)[None, :]
    core = np.real(np.fft.ifft2(1j * ky * np.fft.fft2(psi[:nxc, :nyc])))
    out = np.empty((nxc + 1, nyc + 1), dtype=np.float64)
    out[:nxc, :nyc] = core
    out[nxc, :nyc] = core[0, :]
    out[:nxc, nyc] = core[:, 0]
    out[nxc, nyc] = core[0, 0]
    return out


def solenoidal_residual(Bx, By, psi, dx, dy, nxc, nyc):
    """rms|B_inplane - curl(psi zhat)| / rms|B_inplane|: the compressive fraction
    with no flux function. 0.05 means 5% of the in-plane field is projected out."""
    kx = 2.0 * np.pi * np.fft.fftfreq(nxc, d=dx)[:, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(nyc, d=dy)[None, :]
    psi_k = np.fft.fft2(psi[:nxc, :nyc])
    bx_rec = np.real(np.fft.ifft2(1j * ky * psi_k))
    by_rec = np.real(np.fft.ifft2(-1j * kx * psi_k))
    bx, by = Bx[:nxc, :nyc], By[:nxc, :nyc]
    num = np.sqrt(np.mean((bx - bx_rec)**2 + (by - by_rec)**2))
    den = np.sqrt(np.mean(bx**2 + by**2))
    return float(num / max(den, 1e-300))


def flux_function_path(Bx, By, dx, dy):
    """psi via path integration (kept for --psi-method path). Path-dependent
    whenever div.B != 0 discretely; the FFT solve is preferred."""
    psi_x0 = -_cumtrapz0(By[:, 0], dx, axis=0)
    return psi_x0[:, None] + _cumtrapz0(Bx, dy, axis=1)


def psi_hessian_fft(psi, dx, dy, nxc, nyc):
    """Periodic Hessian of psi on the node grid, consistent with the FFT solve."""
    core = psi[:nxc, :nyc]
    kx = 2.0 * np.pi * np.fft.fftfreq(nxc, d=dx)[:, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(nyc, d=dy)[None, :]
    pk = np.fft.fft2(core)
    hxx = np.real(np.fft.ifft2(-(kx**2) * pk))
    hxy = np.real(np.fft.ifft2(-(kx * ky) * pk))
    hyy = np.real(np.fft.ifft2(-(ky**2) * pk))
    out = []
    for a in (hxx, hxy, hyy):
        q = np.empty((nxc + 1, nyc + 1), dtype=np.float64)
        q[:nxc, :nyc] = a
        q[nxc, :nyc] = a[0, :]
        q[:nxc, nyc] = a[:, 0]
        q[nxc, nyc] = a[0, 0]
        out.append(q)
    return tuple(out)


def find_sheet_centre(prof, lo, hi):
    """Locate one current sheet as the sign change of the along-sheet-averaged
    reversing-field profile within [lo, hi). Steepest crossing wins. The exact-
    zero term catches a sheet sitting exactly on a node. Returns a global y
    index or None. Reversing field is Bx for XY, Bz for YZ -- the caller passes
    whichever profile is appropriate."""
    seg = prof[lo:hi]
    s = np.sign(seg)
    cr = np.where((s[:-1] * s[1:] < 0) | (s[:-1] == 0))[0]
    if cr.size == 0:
        return None
    grad = np.abs(np.diff(seg))
    return lo + int(cr[np.argmax(grad[cr])])


def dpsi_from_neutral_line(Bx, psi, j_lo, j_hi):
    """
    Within the y-band [j_lo, j_hi], find the Bx = 0 crossing per H-column (linear
    interpolation), evaluate psi there, return

        dpsi = max(psi_neutral) - min(psi_neutral)   ( = psi_X - psi_O )

    All crossings per column are kept. Returns (dpsi, count, locs, imax, imin);
    (nan, 0, None, None, None) if fewer than two samples. XY only.
    """
    vals = []
    locs = []
    for i in range(Bx.shape[0]):
        col = Bx[i, j_lo:j_hi + 1]
        s = np.sign(col)
        cross = np.where((s[:-1] * s[1:] < 0) | (s[:-1] == 0))[0]
        for m in cross:
            j = j_lo + m
            b0, b1 = Bx[i, j], Bx[i, j + 1]
            if b0 == b1:
                continue
            w = b0 / (b0 - b1)
            vals.append(psi[i, j] * (1.0 - w) + psi[i, j + 1] * w)
            locs.append((i, j, w))
    if len(vals) < 2:
        return np.nan, 0, None, None, None
    v = np.asarray(vals)
    return (float(v.max() - v.min()), int(v.size), locs,
            int(np.argmax(v)), int(np.argmin(v)))


def sample_at(field, loc):
    """Linear interpolation of `field` at a neutral-line sample (i, j, w)."""
    if loc is None:
        return np.nan
    i, j, w = loc
    return float(field[i, j] * (1.0 - w) + field[i, j + 1] * w)


def ez_at(Ez, locs, idx, nsmooth):
    """Ez at one psi extremum, optionally averaged over nsmooth consecutive
    along-line samples centred there (cuts PIC shot noise as 1/sqrt(N); psi is
    stationary at a critical point so the bias is small). Wraps periodically in
    the along-sheet index. XY only."""
    if locs is None or idx is None:
        return np.nan
    n = len(locs)
    h = max(0, int(nsmooth) // 2)
    if h == 0 or n < 3:
        return sample_at(Ez, locs[idx])
    ids = [(idx + m) % n for m in range(-h, h + 1)]
    return float(np.mean([sample_at(Ez, locs[q]) for q in ids]))


def identify_xo_points(Bx, psi, j_lo, j_hi, dx, dy, nxc, nyc):
    """Identify X/O points on one sheet from the neutral line and psi Hessian.
    One crossing per H-column (strongest local Bx gradient); local psi extrema
    along the line classified by det(H): det<0 -> X, det>0 -> O. Periodic H. XY
    only."""
    hxx, hxy, hyy = psi_hessian_fft(psi, dx, dy, nxc, nyc)
    psi_n = np.full(nxc, np.nan, dtype=np.float64)
    loc_n = [None] * nxc
    for i in range(nxc):
        col = Bx[i, j_lo:j_hi + 1]
        s = np.sign(col)
        cross = np.where((s[:-1] * s[1:] < 0) | (s[:-1] == 0))[0]
        candidates = []
        for m in cross:
            j = j_lo + m
            b0, b1 = Bx[i, j], Bx[i, j + 1]
            if b0 == b1:
                continue
            w = b0 / (b0 - b1)
            candidates.append((abs(b1 - b0), j, w))
        if not candidates:
            continue
        _, j, w = max(candidates, key=lambda q: q[0])
        psi_n[i] = psi[i, j] * (1.0 - w) + psi[i, j + 1] * w
        loc_n[i] = (i, j, w)
    points = []
    for i in range(nxc):
        if not np.isfinite(psi_n[i]):
            continue
        left_i = (i - 1) % nxc
        right_i = (i + 1) % nxc
        if not np.isfinite(psi_n[left_i]) or not np.isfinite(psi_n[right_i]):
            continue
        is_max = psi_n[i] > psi_n[left_i] and psi_n[i] >= psi_n[right_i]
        is_min = psi_n[i] < psi_n[left_i] and psi_n[i] <= psi_n[right_i]
        if not (is_max or is_min):
            continue
        i0, j, w = loc_n[i]
        Hxx = hxx[i0, j] * (1.0 - w) + hxx[i0, j + 1] * w
        Hxy = hxy[i0, j] * (1.0 - w) + hxy[i0, j + 1] * w
        Hyy = hyy[i0, j] * (1.0 - w) + hyy[i0, j + 1] * w
        det = Hxx * Hyy - Hxy * Hxy
        if det < 0.0:
            kind = "X"
        elif det > 0.0:
            kind = "O"
        else:
            continue
        points.append(dict(kind=kind, x=i * dx, y=(j + w) * dy,
                           psi=float(psi_n[i]), loc=loc_n[i], det=float(det)))
    return points


def _periodic_dx(x1, x2, Lx):
    d = abs(x1 - x2)
    return min(d, Lx - d)


def assign_xo_ids(points, sheet, cycle, tracker, Lx, max_move):
    """Keep X/O IDs by nearest-position matching; unmatched get new IDs."""
    for kind in ("X", "O"):
        current = [p for p in points if p["kind"] == kind and p.get("sheet") == sheet]
        previous = tracker["points"].setdefault((sheet, kind), {})
        unused = set(previous.keys())
        for p in sorted(current, key=lambda q: q["x"]):
            best_id = None
            best_dist = np.inf
            for pid in unused:
                old = previous[pid]
                dist = np.hypot(_periodic_dx(p["x"], old["x"], Lx), p["y"] - old["y"])
                if dist < best_dist:
                    best_dist = dist
                    best_id = pid
            if best_id is not None and best_dist <= max_move:
                p["id"] = best_id
                unused.remove(best_id)
            else:
                p["id"] = tracker["next_id"][kind]
                tracker["next_id"][kind] += 1
        tracker["points"][(sheet, kind)] = {
            p["id"]: dict(x=p["x"], y=p["y"], psi=p["psi"], cycle=cycle)
            for p in current
        }
    return points


def assign_xo_ids_all(points_by_sheet, cycle, tracker, Lx, max_move):
    out = []
    for sheet, points in points_by_sheet.items():
        for p in points:
            p["sheet"] = sheet
        assign_xo_ids(points, sheet, cycle, tracker, Lx, max_move)
        out.extend(points)
    return out


def pair_with_ids(points, sheet):
    """Pair each X with both neighboring O points after IDs are assigned."""
    xs = sorted([p for p in points if p["kind"] == "X"], key=lambda p: p["x"])
    os_ = sorted([p for p in points if p["kind"] == "O"], key=lambda p: p["x"])
    if not xs or not os_:
        return []
    pairs = []
    for xpt in xs:
        left = [o for o in os_ if o["x"] < xpt["x"]]
        right = [o for o in os_ if o["x"] > xpt["x"]]
        ol = left[-1] if left else os_[-1]
        or_ = right[0] if right else os_[0]
        for side, opt in (("L", ol), ("R", or_)):
            pair_id = f"X{xpt['id']}-O{opt['id']}-{side}"
            pairs.append(dict(sheet=sheet, side=side, pair_id=pair_id,
                              X=xpt, O=opt,
                              dpsi=float(xpt["psi"] - opt["psi"])))
    return pairs


def update_pair_rates(pairs, tracker, t):
    """Differentiate signed dpsi per persistent X-O pair; new pairs get NaN."""
    out = []
    previous = tracker["pairs"]
    for pair in pairs:
        pid = pair["pair_id"]
        dpsi = pair["dpsi"]
        if pid in previous and pid in tracker["active_pairs"]:
            old = previous[pid]
            dt = t - old["t"]
            rate = (dpsi - old["dpsi"]) / dt if dt > 0 else np.nan
        else:
            rate = np.nan
        pair["ddpsi_dt"] = float(rate)
        previous[pid] = dict(dpsi=dpsi, t=t)
        out.append(pair)
    tracker["active_pairs"] = {p["pair_id"] for p in pairs}
    return out


def measure_B0(Bx, c1, c2, nyg):
    """Upstream |<Bx>_H| midway between the sheets, as a cross-check on --B0.
    DIAGNOSTIC ONLY. Bx here means the reversing field (Bx for XY)."""
    m_in = (c1 + c2) // 2
    m_out = ((c2 + c1 + nyg) // 2) % nyg
    return 0.5 * (float(np.abs(Bx[:, m_in]).mean()) +
                  float(np.abs(Bx[:, m_out]).mean()))


###! ============================================================
###! Paper-method inflow/outflow sampling  (both planes; inflow always Vy)
###! ============================================================

def vrec_paper_one_sheet(Vy, Vout, c, quarter, nyg, nHg, LH_cells):
    """
    Paper-method (Appendix G) inflow/outflow sampling for ONE sheet centred at
    global y-node `c`. Inflow is ALWAYS Vy; the outflow component `Vout` is Vx
    for XY and Vz for YZ -- the caller passes whichever it is. The along-sheet
    axis is H (x for XY, z for YZ) and is the FIRST axis of Vy and Vout.

      v_in  : mean of INFLOWING Vy over the upstream RECTANGLE
                  INFLOW_FRAC_LO*quarter <= |y-c| <= INFLOW_FRAC_HI*quarter  (y)
              AND INFLOW_H_LO*LH <= H <= INFLOW_H_HI*LH                       (H)
              on BOTH y-sides. Inflowing = Vy directed toward c: Vy>0 for y<c,
              Vy<0 for y>c. Only inflowing-sign samples averaged (magnitudes).
      v_out : max |Vout| within |y-c| <= OUTFLOW_HALF_FRAC*quarter, over the
              H-range with OUTFLOW_H_GUARD*LH removed from each end. H is
              periodic, so the guard removes no boundary artifact.

    `quarter` = Ly/4 in NODES. Returns (v_in, v_out, ratio, n_in). ratio =
    v_in/v_out. For XY this ratio IS the reconnection v_rec; for YZ it is a FLOW
    ratio only and the caller labels it as such.
    """
    j_in_lo = int(round(INFLOW_FRAC_LO * quarter))
    j_in_hi = int(round(INFLOW_FRAC_HI * quarter))
    j_out   = int(round(OUTFLOW_HALF_FRAC * quarter))
    i_guard = int(round(OUTFLOW_H_GUARD * LH_cells))

    y = np.arange(nyg)
    off = y - c

    ih_lo = int(round(INFLOW_H_LO * LH_cells))
    ih_hi = int(round(INFLOW_H_HI * LH_cells))
    if ih_hi <= ih_lo:
        ih_lo, ih_hi = 0, nHg - 1

    band_lo = (off <= -j_in_lo) & (off >= -j_in_hi)     ###! low-y side (y < c)
    band_hi = (off >=  j_in_lo) & (off <=  j_in_hi)     ###! high-y side (y > c)

    vin_samples = []
    if band_lo.any():
        blk = Vy[ih_lo:ih_hi + 1, band_lo]              ###! inflow here is Vy > 0
        vin_samples.append(blk[blk > 0.0])
    if band_hi.any():
        blk = Vy[ih_lo:ih_hi + 1, band_hi]              ###! inflow here is Vy < 0
        vin_samples.append(-blk[blk < 0.0])
    allin = np.concatenate(vin_samples) if vin_samples else np.empty(0)
    n_in = int(allin.size)
    v_in = float(allin.mean()) if n_in > 0 else np.nan

    band_out = np.abs(off) <= j_out
    h_hi = (nHg - 1) - i_guard
    if 2 * i_guard >= nHg - 1:
        h_lo, h_hi = 0, nHg - 1
    else:
        h_lo = i_guard
    if band_out.any():
        v_out = float(np.abs(Vout[h_lo:h_hi + 1, band_out]).max())
    else:
        v_out = np.nan

    ratio = v_in / v_out if (np.isfinite(v_out) and v_out > 0.0) else np.nan
    return v_in, v_out, ratio, n_in


###! ============================================================
###! Arguments
###! ============================================================

p = argparse.ArgumentParser(
    description="Reconnection rate (XY) or inflow/outflow FLOW diagnostic (YZ) "
                "for a 2D double-Harris iPIC3D run. Reads a single resolved "
                "plane from a run whose third axis is collapsed to one tile.")

p.add_argument("dir_data", type=str, help="Directory containing proc*.hdf")
p.add_argument("outdir",   type=str, help="Output directory")
p.add_argument("xmin", type=float); p.add_argument("xmax", type=float)
p.add_argument("ymin", type=float); p.add_argument("ymax", type=float)
p.add_argument("zmin", type=float); p.add_argument("zmax", type=float)

p.add_argument("--plane", type=str, required=True, choices=["xy", "yz"],
               help="Resolved plane. xy: sheet along x, collapse z, outflow Vx, "
                    "reconnection MEASURED. yz: sheet along z, collapse x, "
                    "outflow Vz, FLOW DIAGNOSTIC ONLY (no reconnection).")

p.add_argument("--nxc", type=int, required=True, help="Number of cells in x")
p.add_argument("--nyc", type=int, required=True, help="Number of cells in y")
p.add_argument("--nzc", type=int, required=True, help="Number of cells in z")
p.add_argument("--xlen", type=int, default=None, help="Override derived XLEN")
p.add_argument("--ylen", type=int, default=None, help="Override derived YLEN")
p.add_argument("--zlen", type=int, default=None, help="Override derived ZLEN")

p.add_argument("--sigma", type=float, required=True,
               help="Ion magnetisation sigma_i. Sets vA via sqrt(s/(1+s)). Used "
                    "for the XY flux-rate normalisation; unused for the YZ flow "
                    "diagnostic (which is not normalised by vA).")
p.add_argument("--guide-field", type=float, default=0.0,
               help="Bz/B0 for the outflow enthalpy loading of vA (XY only): "
                    "vA = c sqrt(s/(1+s+s_g)), s_g=(Bz/B0)^2 s. Default 0.")
p.add_argument("--mass-ratio", type=float, default=1836.0,
               help="m_i/m_e. Sets the enthalpy correction to vA AND the per-"
                    "species mass weighting for the bulk velocity (weight -1 for "
                    "electrons 0,2; +mass_ratio for protons 1,3).")
p.add_argument("--theta-i", type=float, default=None,
               help="Upstream ion thermal spread (enthalpy correction to vA).")
p.add_argument("--theta-e", type=float, default=None,
               help="Upstream electron thermal spread (enthalpy correction).")
p.add_argument("--time-denom", type=float, required=True,
               help="t*omega_p = cycle / time_denom.")
p.add_argument("--B0", type=float, default=None,
               help="Asymptotic upstream |B_reversing|. If omitted, measured from "
                    "the first valid dump midway between the sheets (XY only; the "
                    "YZ flow diagnostic does not use B0).")

p.add_argument("--no-vrec-paper", dest="vrec_paper", action="store_false",
               help="XY: skip the paper-method v_rec. (YZ: the flow diagnostic IS "
                    "the paper-method sampling, so this also skips the only YZ "
                    "output; a warning is printed.)")

p.add_argument("--cycle-start", type=int, default=0)
p.add_argument("--cycle-end",   type=int, default=20000)
p.add_argument("--cycle-step",  type=int, default=500)
p.add_argument("--cycle-chunk", type=int, default=5,
               help="Cycles held in memory at once. Memory ~ chunk * n_datasets "
                    "* GH * Gy * 8 bytes per rank.")

p.add_argument("--band-half-width", type=int, default=None,
               help="XY: restrict the neutral-line search to +/- this many cells "
                    "around each sheet centre. Default: use each y-half.")
p.add_argument("--check-xo", action="store_true",
               help="XY: write xo_check.txt verifying the psi extrema are X/O "
                    "points. Ignored for YZ.")
p.add_argument("--ez-smooth", type=int, default=1,
               help="XY: average Ez over this many along-line samples around each "
                    "psi extremum before the E-based rate. 1 = no smoothing.")
p.add_argument("--psi-method", type=str, default="fft", choices=["fft", "path"],
               help="XY: how to build psi. 'fft' (default) is the spectral "
                    "Poisson solve, valid for the fully periodic in-plane field.")
p.add_argument("--no-plot", dest="plot", action="store_false",
               help="Skip the .png figures; write only the .txt tables.")
p.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"])
p.add_argument("--mapping", type=str, default="A",
               choices=["auto", "A", "B", "C", "D", "E", "F"],
               help="proc->(i,j,k) mapping. With a collapsed axis several "
                    "candidates tie under 'auto'; if the auto pick is ever wrong "
                    "for your file ordering, force it here.")

args = p.parse_args()

work_dtype = np.float32 if args.dtype == "float32" else np.float64
is_xy = (args.plane == "xy")


###! ============================================================
###! Geometry
###! ============================================================

nxc, nyc, nzc = args.nxc, args.nyc, args.nzc
Lx = args.xmax - args.xmin
Ly = args.ymax - args.ymin
Lz = args.zmax - args.zmin
for lbl, L, lo, hi in (("x", Lx, args.xmin, args.xmax),
                       ("y", Ly, args.ymin, args.ymax),
                       ("z", Lz, args.zmin, args.zmax)):
    if L <= 0:
        raise SystemExit(f"Box length in {lbl} is {L} (from {lbl}min={lo}, "
                         f"{lbl}max={hi}). Extents must satisfy max > min.")

nxg, nyg, nzg = nxc + 1, nyc + 1, nzc + 1
dx, dy, dz = Lx / nxc, Ly / nyc, Lz / nzc
quarter_nodes = nyc / 4.0

###! resolved horizontal (along-sheet) axis H, its length, cell count, and the
###! outflow current/velocity component that lives along it.
if is_xy:
    LH, nHc, h_label = Lx, nxc, "x"
    comp_out = "Jx"                          ###! outflow current component
    reversing = "Bx"                         ###! sheet reverses in Bx across y
else:
    LH, nHc, h_label = Lz, nzc, "z"
    comp_out = "Jz"
    reversing = "Bz"                         ###! sheet reverses in Bz across y
nHg = nHc + 1

###! collapsed axis validation (mode check, mirrors Spectra_compute.py)
ncs = (nxc, nyc, nzc)
ca = COLL_AXIS_OF[args.plane]
mode_error = None
if ncs[ca] != 1:
    mode_error = (f"--plane {args.plane} collapses {'xyz'[ca]}, but "
                  f"n{'xyz'[ca]}c = {ncs[ca]} (expected 1 for a 2D-spatial run).")
elif ncs[1] == 1:
    mode_error = "y is collapsed (nyc == 1); this diagnostic needs y resolved."
else:
    for a in range(3):
        if a != ca and ncs[a] == 1:
            mode_error = (f"--plane {args.plane} but n{'xyz'[a]}c = 1 as well: "
                          f"more than one axis is collapsed, not a 2D plane.")
            break
if mode_error is not None:
    if rank == 0:
        print(f"MODE ERROR: {mode_error}", flush=True)
    comm.Barrier(); raise SystemExit(1)


def mean_lorentz(theta):
    return 1.0 + theta * (6.0 + 15.0 * theta) / (4.0 + 5.0 * theta)


###! effective magnetisation and vA (XY flux-rate normalisation)
if args.mass_ratio is None:
    sigma_eff = args.sigma
    vA_note = "cold ion rest-mass only"
else:
    g_i = mean_lorentz(args.theta_i) if args.theta_i is not None else 1.0
    g_e = mean_lorentz(args.theta_e) if args.theta_e is not None else 1.0
    sigma_eff = args.sigma / (g_i + g_e / args.mass_ratio)
    vA_note = (f"enthalpy-corrected: R={args.mass_ratio:g}, "
               f"<g_i>={g_i:.4f}, <g_e>={g_e:.4f}")
sigma_g = (args.guide_field ** 2) * sigma_eff
vA = np.sqrt(sigma_eff / (1.0 + sigma_eff + sigma_g))
if args.guide_field != 0.0:
    vA_note += (f"; guide-field loaded: b=Bz/B0={args.guide_field:g}, "
                f"sigma_g={sigma_g:.6f}")


###! ============================================================
###! Discover files, probe tile shape, per-species availability, mapping
###! ============================================================

if rank == 0:
    all_files = sorted(glob.glob(os.path.join(args.dir_data, "proc*.hdf")))
    if not all_files:
        raise RuntimeError(f"No proc*.hdf found in {args.dir_data}")

    first_cycle = f"cycle_{args.cycle_start}"
    with h5py.File(all_files[0], "r") as f:
        ###! reversing-field and normal-field components must both exist
        pRev = f"fields/{reversing}/{first_cycle}"
        pBy  = f"fields/By/{first_cycle}"
        if pRev not in f:
            raise KeyError(f"Missing dataset {pRev} in {all_files[0]}")
        if pBy not in f:
            raise KeyError(f"Missing dataset {pBy} in {all_files[0]}")
        tile_shape = tuple(f[pRev].shape)
        if tuple(f[pBy].shape) != tile_shape:
            raise RuntimeError(f"{reversing} and By have different tile shapes; "
                               "fields are not colocated.")
        ###! Ez optional (XY E-based rate). YZ never uses it.
        HAVE_EZ = is_xy and (f"fields/Ez/{first_cycle}" in f)
        if HAVE_EZ and tuple(f[f"fields/Ez/{first_cycle}"].shape) != tile_shape:
            print("  WARNING: Ez tile shape differs; disabling E-based rate.",
                  flush=True)
            HAVE_EZ = False

        ###! per-species moments: need rho, Jy, and the outflow component comp_out
        HAVE_SPECIES = True
        species_probe = None
        for s in ALL_SPECIES:
            for q in ("rho", "Jy", comp_out):
                pth = f"moments/species_{s}/{q}/{first_cycle}"
                if pth not in f:
                    HAVE_SPECIES = False
                    species_probe = pth
                    break
            if not HAVE_SPECIES:
                break
        if HAVE_SPECIES:
            ms = tuple(f[f"moments/species_0/rho/{first_cycle}"].shape)
            if ms != tile_shape:
                raise RuntimeError(f"Moment tile shape {ms} != field tile shape "
                                   f"{tile_shape}; shared assembly indexing wrong.")

    nx_t, ny_t, nz_t = tile_shape
    for lbl, n_c, n_t in (("x", nxc, nx_t), ("y", nyc, ny_t), ("z", nzc, nz_t)):
        if n_t < 2 or n_c % (n_t - 1) != 0:
            raise RuntimeError(
                f"Cannot derive the {lbl} decomposition: {n_c} cells do not "
                f"divide into tiles of {n_t - 1} cells (tile {tile_shape}).")
    XLEN = args.xlen if args.xlen is not None else nxc // (nx_t - 1)
    YLEN = args.ylen if args.ylen is not None else nyc // (ny_t - 1)
    ZLEN = args.zlen if args.zlen is not None else nzc // (nz_t - 1)
    if XLEN * YLEN * ZLEN != len(all_files):
        raise RuntimeError(
            f"Derived decomposition {XLEN}x{YLEN}x{ZLEN} = {XLEN*YLEN*ZLEN} "
            f"tiles, but {len(all_files)} proc*.hdf present.")
    ###! the collapsed axis MUST be a single tile, or "sample the lower node" is
    ###! wrong (Spectra_compute.py imposes the same).
    coll_len = ZLEN if is_xy else XLEN
    if coll_len != 1:
        raise RuntimeError(
            f"Collapsed axis has {coll_len} tiles (expected 1 for plane "
            f"{args.plane}). A 2D run must not decompose the collapsed axis.")

    if args.mapping == "auto":
        map_name = choose_mapping(all_files, XLEN, YLEN, ZLEN)
        print(f"Auto-selected mapping '{map_name}' (collapsed axis makes several "
              f"tie; pass --mapping if wrong).", flush=True)
    else:
        map_name = args.mapping

    print(f"Plane           : {args.plane.upper()}  (H = {h_label}, along-sheet; "
          f"normal = y; collapse {'xyz'[ca]})", flush=True)
    print(f"Reconnection    : {'MEASURED (flux + paper v_rec)' if is_xy else 'NONE -- flow diagnostic only'}",
          flush=True)
    print(f"Decomposition   : {XLEN} x {YLEN} x {ZLEN} = {XLEN*YLEN*ZLEN} files",
          flush=True)
    print(f"Per-species mom : {'present' if HAVE_SPECIES else 'ABSENT'}"
          f"{'' if HAVE_SPECIES else '  (' + str(species_probe) + ' missing)'}",
          flush=True)
else:
    all_files = tile_shape = map_name = None
    XLEN = YLEN = ZLEN = None
    HAVE_EZ = HAVE_SPECIES = None

all_files    = comm.bcast(all_files, root=0)
tile_shape   = comm.bcast(tile_shape, root=0)
map_name     = comm.bcast(map_name, root=0)
XLEN         = comm.bcast(XLEN, root=0)
YLEN         = comm.bcast(YLEN, root=0)
ZLEN         = comm.bcast(ZLEN, root=0)
HAVE_EZ      = comm.bcast(HAVE_EZ, root=0)
HAVE_SPECIES = comm.bcast(HAVE_SPECIES, root=0)

rank_to_ijk = mapping_candidates(XLEN, YLEN, ZLEN)[map_name]
local_files = all_files[rank::size]
G_shape = (XLEN * (tile_shape[0] - 1) + 1,
           YLEN * (tile_shape[1] - 1) + 1,
           ZLEN * (tile_shape[2] - 1) + 1)
coll_index = 0

###! do we attempt the paper/flow sampling at all?
do_paper = bool(args.vrec_paper and HAVE_SPECIES)
if rank == 0 and (not is_xy) and (not do_paper):
    if not HAVE_SPECIES:
        print("YZ FLOW DIAGNOSTIC: per-species moments absent -- NOTHING to "
              "write for YZ (the flow diagnostic IS the paper sampling).",
              flush=True)
    else:
        print("YZ FLOW DIAGNOSTIC: --no-vrec-paper given -- nothing to write.",
              flush=True)

###! datasets to assemble. XY needs Bx,By(,Ez) for the flux rate; YZ needs NO
###! fields for the (nonexistent) flux rate but DOES need Bz to locate the sheet
###! for the flow sampling. Both need the per-species moments IF do_paper.
DATASETS = {}
if is_xy:
    DATASETS["Bx"] = "fields/Bx/{cycle}"
    DATASETS["By"] = "fields/By/{cycle}"
    if HAVE_EZ:
        DATASETS["Ez"] = "fields/Ez/{cycle}"
else:
    ###! YZ: only the reversing field is needed, to find the sheet centre for the
    ###! flow sampling. No flux function is built.
    DATASETS["Bz"] = "fields/Bz/{cycle}"
if do_paper:
    for s in ALL_SPECIES:
        DATASETS[f"rho_s{s}"] = f"moments/species_{s}/rho/{{cycle}}"
        DATASETS[f"Jy_s{s}"]  = f"moments/species_{s}/Jy/{{cycle}}"
        DATASETS[f"{comp_out}_s{s}"] = f"moments/species_{s}/{comp_out}/{{cycle}}"

if rank == 0:
    print(f"Tile shape      : {tile_shape}", flush=True)
    print(f"Cells           : {nxc} x {nyc} x {nzc}   Nodes: {nxg} x {nyg} x {nzg}",
          flush=True)
    print(f"Box lengths     : Lx={Lx:.6g} Ly={Ly:.6g} Lz={Lz:.6g}", flush=True)
    print(f"Spacing         : dx={dx:.6g} dy={dy:.6g} dz={dz:.6g}", flush=True)
    print(f"Sheet reverses  : {reversing} across y ; outflow = V{h_label} "
          f"(from {comp_out})", flush=True)
    if is_xy:
        print(f"vA/c            : {vA:.6f}  (sigma_i={args.sigma:g}, "
              f"sigma_eff={sigma_eff:.6f}); {vA_note}", flush=True)
    print(f"Datasets read   : {list(DATASETS.keys())}", flush=True)


###! ============================================================
###! XY per-plane analysis (flux function + neutral line + X/O)
###! ============================================================

def analyse_xy(Bx, By, Ez, xo_tracker, cycle):
    """Full flux-function analysis of the single XY plane. Returns a record dict
    or None if both sheets could not be located. XY ONLY."""
    if args.psi_method == "fft":
        psi = flux_function_fft(Bx, By, dx, dy, nxc, nyc)
    else:
        psi = flux_function_path(Bx, By, dx, dy)

    resid = solenoidal_residual(Bx, By, psi, dx, dy, nxc, nyc)
    Bx_search = (bx_from_psi(psi, dy, nxc, nyc)
                 if args.psi_method == "fft" else Bx)

    prof = Bx_search.mean(axis=0)
    mid = nyg // 2
    c1 = find_sheet_centre(prof, 0, mid)
    c2 = find_sheet_centre(prof, mid, nyg)
    if c1 is None or c2 is None:
        return None

    if args.band_half_width is None:
        b1 = (0, mid - 1)
        b2 = (mid, nyg - 1)
    else:
        h = args.band_half_width
        b1 = (max(0, c1 - h), min(mid - 1, c1 + h))
        b2 = (max(mid, c2 - h), min(nyg - 1, c2 + h))

    d1, n1, L1, imax1, imin1 = dpsi_from_neutral_line(Bx_search, psi, *b1)
    d2, n2, L2, imax2, imin2 = dpsi_from_neutral_line(Bx_search, psi, *b2)

    if Ez is None:
        e1 = e2 = np.nan
    else:
        ns = args.ez_smooth
        e1 = -(ez_at(Ez, L1, imax1, ns) - ez_at(Ez, L1, imin1, ns))   ###! c = 1
        e2 = -(ez_at(Ez, L2, imax2, ns) - ez_at(Ez, L2, imin2, ns))

    xo = {}
    if args.check_xo:
        By_sol = -np.gradient(psi, dx, axis=0) if args.psi_method == "fft" else By
        for tag, L_, imx, imn in (("1", L1, imax1, imin1), ("2", L2, imax2, imin2)):
            if L_ is None:
                continue
            byline = np.array([abs(sample_at(By_sol, q)) for q in L_])
            typ = float(np.mean(byline)) if byline.size else np.nan
            xo[f"bymax{tag}"] = byline[imx] / typ if typ > 0 else np.nan
            xo[f"bymin{tag}"] = byline[imn] / typ if typ > 0 else np.nan
            xo[f"xmax{tag}"] = (L_[imx][0] + L_[imx][2]) * dx
            xo[f"xmin{tag}"] = (L_[imn][0] + L_[imn][2]) * dx
            if Ez is not None:
                xo[f"ezmax{tag}"] = ez_at(Ez, L_, imx, args.ez_smooth)
                xo[f"ezmin{tag}"] = ez_at(Ez, L_, imn, args.ez_smooth)

    xo_points = []
    xo_pairs = []
    if xo_tracker is not None:
        points_by_sheet = {
            1: identify_xo_points(Bx_search, psi, *b1, dx, dy, nxc, nyc),
            2: identify_xo_points(Bx_search, psi, *b2, dx, dy, nxc, nyc),
        }
        xo_points = assign_xo_ids_all(points_by_sheet, cycle, xo_tracker,
                                       Lx, xo_tracker["max_move"])
        pairs = []
        for sheet, sheet_points in points_by_sheet.items():
            pairs.extend(pair_with_ids(sheet_points, sheet))
        xo_pairs = update_pair_rates(pairs, xo_tracker, cycle / args.time_denom)

    return dict(y1=c1, y2=c2, d1=d1, d2=d2, n1=n1, n2=n2, pi=resid,
                e1=e1, e2=e2, xo=xo, xo_points=xo_points, xo_pairs=xo_pairs,
                B0m=measure_B0(Bx, c1, c2, nyg))


###! ============================================================
###! Main loop over cycle chunks
###! ============================================================

cycles_all = list(range(args.cycle_start, args.cycle_end + 1, args.cycle_step))
xy_records = []          ###! rank 0: XY flux records
xo_pair_records = []     ###! rank 0
paper_records = []       ###! rank 0: paper/flow records (both planes)
B0 = args.B0
xo_tracker = {"points": {}, "next_id": {"X": 0, "O": 0}, "pairs": {},
              "active_pairs": set(),
              "max_move": max(8.0 * dx, 0.02 * Lx)} if is_xy else None

for c0 in range(0, len(cycles_all), args.cycle_chunk):
    chunk = cycles_all[c0:c0 + args.cycle_chunk]
    names = [f"cycle_{c}" for c in chunk]
    if rank == 0:
        print(f"\nAssembling {names[0]} .. {names[-1]}", flush=True)

    out = assemble_plane_chunk(names, local_files, rank_to_ijk, tile_shape,
                               G_shape, DATASETS, args.plane, coll_index,
                               work_dtype)
    if rank != 0:
        continue

    fields = out["fields"]
    for ci, cyc in enumerate(chunk):
        if out["found"][ci] <= 0:
            why = "dataset absent" if out["found"][ci] == 0 else "incomplete across tiles"
            print(f"  cycle_{cyc}: {why}, skipped", flush=True)
            continue

        ###! ----- locate the sheets (both planes need centres) -----
        if is_xy:
            rec = analyse_xy(fields["Bx"][ci], fields["By"][ci],
                             fields["Ez"][ci] if "Ez" in fields else None,
                             xo_tracker, cyc)
            if rec is None:
                print(f"  cycle_{cyc}: could not locate both sheets, skipped",
                      flush=True)
                continue
            if B0 is None:
                B0 = rec["B0m"]
                print(f"  B0 measured midway between sheets: {B0:.6e}", flush=True)
            elif not xy_records and abs(rec["B0m"] - B0) > 0.1 * abs(B0):
                print(f"  WARNING: --B0={B0:.4e} but measured |Bx|={rec['B0m']:.4e} "
                      f"(>10% off).", flush=True)
            rec.update(cycle=cyc, t=cyc / args.time_denom)
            xy_records.append(rec)
            c1, c2 = rec["y1"], rec["y2"]
            for pair in rec.get("xo_pairs", []):
                xo_pair_records.append(dict(
                    cycle=cyc, t=cyc / args.time_denom, sheet=pair["sheet"],
                    side=pair["side"], pair_id=pair["pair_id"],
                    x_id=pair["X"]["id"], o_id=pair["O"]["id"],
                    x_x=pair["X"]["x"], x_y=pair["X"]["y"],
                    o_x=pair["O"]["x"], o_y=pair["O"]["y"],
                    psi_x=pair["X"]["psi"], psi_o=pair["O"]["psi"],
                    dpsi=pair["dpsi"], ddpsi_dt=pair["ddpsi_dt"]))
        else:
            ###! YZ: no flux function. Locate the sheets from <Bz>_z(y).
            Bz = fields["Bz"][ci]
            prof = Bz.mean(axis=0)
            mid = nyg // 2
            c1 = find_sheet_centre(prof, 0, mid)
            c2 = find_sheet_centre(prof, mid, nyg)
            if c1 is None or c2 is None:
                print(f"  cycle_{cyc}: could not locate both sheets in <Bz>_z, "
                      f"skipped", flush=True)
                continue

        ###! ----- paper / flow sampling (both planes; inflow always Vy) -----
        if do_paper:
            Vy, Vout = build_velocity(fields, ci, args.mass_ratio, comp_out)
            vin1, vout1, r1, nin1 = vrec_paper_one_sheet(
                Vy, Vout, c1, quarter_nodes, nyg, nHg, nHc)
            vin2, vout2, r2, nin2 = vrec_paper_one_sheet(
                Vy, Vout, c2, quarter_nodes, nyg, nHg, nHc)
            paper_records.append(dict(
                cycle=cyc, t=cyc / args.time_denom, y1=c1, y2=c2,
                vin1=vin1, vout1=vout1, ratio1=r1, nin1=nin1,
                vin2=vin2, vout2=vout2, ratio2=r2, nin2=nin2))


###! ============================================================
###! Writers and plots  (rank 0 only)
###! ============================================================

def make_flux_plot(d, path, what):
    """XY flux-rate figure: CS1/CS2 from B, and from Ez if available."""
    if not HAVE_MPL:
        return None
    has_E = "E1" in d and np.any(np.isfinite(d["E1"]))
    if has_E:
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), sharey=True)
        panels = [(axes[0], d["R1"], d["R2"], r"from $\mathbf{B}$:  $d(\Delta\psi)/dt$"),
                  (axes[1], d["E1"], d["E2"], r"from $E_z$:  $-c[E_z^{X}-E_z^{O}]$")]
    else:
        fig, ax1 = plt.subplots(figsize=(8.5, 4.8))
        axes = [ax1]
        panels = [(ax1, d["R1"], d["R2"], None)]
    for a, y1, y2, sub in panels:
        a.axhline(0.0, color="k", lw=0.8)
        a.plot(d["t"], y1, "-", color="C0", lw=1.6, label="CS1")
        a.plot(d["t"], y2, "-", color="C3", lw=1.6, label="CS2")
        a.set_xlabel(r"$t\,\omega_p$", fontsize=13)
        a.grid(alpha=0.3); a.legend(fontsize=11)
        if sub:
            a.set_title(sub, fontsize=12)
    axes[0].set_ylabel(r"$R = \dot{\Delta\psi}\,/\,(B_0 v_A)$", fontsize=13)
    (fig.suptitle if has_E else axes[0].set_title)(what, fontsize=12)
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def make_paper_plot(d, path, is_flow):
    """Paper v_rec (XY) or flow ratio (YZ). Title states which."""
    if not HAVE_MPL:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axhline(0.0, color="k", lw=0.8)
    ax.plot(d["t"], d["r1"], "-", color="C0", lw=1.6, label="CS1")
    ax.plot(d["t"], d["r2"], "-", color="C3", lw=1.6, label="CS2")
    ax.set_xlabel(r"$t\,\omega_p^{-1}$", fontsize=13)
    if is_flow:
        ax.set_ylabel(r"$\langle v_{\rm in}\rangle / v_{\rm out}$  (flow ratio, "
                      r"$v_{\rm out}=|v_z|_{\max}$)", fontsize=12)
        ax.set_title("YZ FLOW DIAGNOSTIC -- NOT a reconnection rate", fontsize=12)
    else:
        ax.set_ylabel(r"$v_{\rm rec} = \langle v_{\rm in}\rangle / v_{\rm out}$",
                      fontsize=13)
    ax.legend(fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def write_flux_table(recs, path, what):
    """XY: differentiate Delta psi in time, write the flux-rate table."""
    recs.sort(key=lambda r: r["cycle"])
    t  = np.array([r["t"]  for r in recs])
    d1 = np.array([r["d1"] for r in recs])
    d2 = np.array([r["d2"] for r in recs])
    R1, R2 = np.gradient(d1, t) / norm, np.gradient(d2, t) / norm
    E1 = np.array([r.get("e1", np.nan) for r in recs]) / norm
    E2 = np.array([r.get("e2", np.nan) for r in recs]) / norm
    pi_max = max(r["pi"] for r in recs)
    with open(path, "w") as fh:
        fh.write(f"# Reconnection rate, 2D {args.plane.upper()} plane -- {what}\n")
        fh.write(f"# dir            = {args.dir_data}\n")
        fh.write(f"# plane          = {args.plane.upper()}  (H={h_label} along sheet; "
                 f"normal=y; reversing field {reversing}; collapse {'xyz'[ca]})\n")
        fh.write(f"# XLEN,YLEN,ZLEN = {XLEN},{YLEN},{ZLEN}   mapping = {map_name}\n")
        fh.write(f"# cells          = {nxc} x {nyc} x {nzc}\n")
        fh.write(f"# extents        = x[{args.xmin},{args.xmax}] "
                 f"y[{args.ymin},{args.ymax}] z[{args.zmin},{args.zmax}]\n")
        fh.write(f"# spacing        = dx={dx:.8g} dy={dy:.8g} dz={dz:.8g}\n")
        fh.write(f"# time           = cycle / {args.time_denom}   [omega_p^-1]\n")
        fh.write(f"# sigma_i={args.sigma}  sigma_eff={sigma_eff:.8f}  vA/c={vA:.8f}\n")
        fh.write(f"# vA basis       = {vA_note}\n")
        fh.write(f"# B0={B0:.8e}  norm=B0*vA={norm:.8e}\n")
        fh.write(f"# psi method     = {args.psi_method}\n")
        fh.write(f"# Ez smoothing   = {args.ez_smooth} neutral-line samples\n")
        fh.write(f"# max resid over all dumps = {pi_max:.3e}\n")
        fh.write("#\n")
        fh.write("# dpsi_CS*  psi_X - psi_O (max-min of psi along Bx=0 line)\n")
        fh.write("# R_CS*     = (1/(B0 vA)) d(dpsi)/dt. SIGN IS MEANINGFUL.\n")
        fh.write("# E_CS*     independent Ez-based rate; should agree with R_CS*\n")
        fh.write("# resid     compressive fraction of B with no flux function\n")
        fh.write("# npts*     Bx=0 crossings (~nHg means one clean crossing/column)\n")
        fh.write("#\n")
        fh.write("# {:>8s} {:>11s} {:>6s} {:>6s} {:>15s} {:>15s} {:>14s} {:>14s} "
                 "{:>11s} {:>7s} {:>7s} {:>14s} {:>14s}\n".format(
                     "cycle", "time", "y_cs1", "y_cs2", "dpsi_CS1", "dpsi_CS2",
                     "R_CS1", "R_CS2", "resid", "npts1", "npts2", "E_CS1", "E_CS2"))
        for i, r in enumerate(recs):
            fh.write("  {:>8d} {:>11.4f} {:>6d} {:>6d} {:>15.7e} {:>15.7e} "
                     "{:>14.6e} {:>14.6e} {:>11.3e} {:>7d} {:>7d} "
                     "{:>14.6e} {:>14.6e}\n".format(
                         r["cycle"], r["t"], r["y1"], r["y2"], d1[i], d2[i],
                         R1[i], R2[i], r["pi"], r["n1"], r["n2"], E1[i], E2[i]))
    return dict(t=t, d1=d1, d2=d2, R1=R1, R2=R2, E1=E1, E2=E2)


def write_paper_table(recs, path, is_flow):
    """Paper-method table. is_flow=False -> XY reconnection v_rec.
    is_flow=True -> YZ flow diagnostic, header states it is NOT a rate."""
    recs.sort(key=lambda r: r["cycle"])
    t  = np.array([r["t"] for r in recs])
    r1 = np.array([r["ratio1"] for r in recs])
    r2 = np.array([r["ratio2"] for r in recs])
    with open(path, "w") as fh:
        if is_flow:
            fh.write("# ============================================================\n")
            fh.write("# YZ-PLANE FLOW DIAGNOSTIC -- NOT A RECONNECTION RATE\n")
            fh.write("# ============================================================\n")
            fh.write("# There is NO reconnection in the YZ plane. This file reports\n")
            fh.write("# an inflow/outflow FLOW characterisation only: inflow = Vy,\n")
            fh.write("# outflow = Vz (max|Vz| in the central band). The ratio\n")
            fh.write("# v_in/v_out is given for completeness and MUST NOT be read\n")
            fh.write("# as a reconnection rate -- there is no X-line here. Use it\n")
            fh.write("# for drift-kink / outflow characterisation.\n")
            fh.write("#\n")
            outname = "Vz"
        else:
            fh.write("# Reconnection rate -- PAPER METHOD (Appendix G): "
                     "v_rec = <v_in>/v_out\n#\n")
            outname = "Vx"
        fh.write(f"# dir            = {args.dir_data}\n")
        fh.write(f"# plane          = {args.plane.upper()}  (H={h_label}; normal=y; "
                 f"sheet reverses in {reversing})\n")
        fh.write(f"# cells          = {nxc} x {nyc} x {nzc}   mapping = {map_name}\n")
        fh.write(f"# extents        = x[{args.xmin},{args.xmax}] "
                 f"y[{args.ymin},{args.ymax}] z[{args.zmin},{args.zmax}]\n")
        fh.write(f"# time           = cycle / {args.time_denom}   [omega_p^-1]\n")
        fh.write(f"# mass_ratio     = {args.mass_ratio:g}  "
                 f"(electrons {SPECIES_ELECTRON}, protons {SPECIES_PROTON})\n")
        fh.write("# velocity       = mass-weighted single-fluid "
                 "V = sum_s (m_s/q_s) J_s / sum_s (m_s/q_s) rho_s\n")
        fh.write(f"# inflow box     = y: {INFLOW_FRAC_LO:g}..{INFLOW_FRAC_HI:g} x (Ly/4) "
                 f"both sides; {h_label}: {INFLOW_H_LO:g}..{INFLOW_H_HI:g} x L{h_label}; "
                 f"inflowing Vy only (mean)\n")
        fh.write(f"# outflow band   = |y - y_cs| <= {OUTFLOW_HALF_FRAC:g} x (Ly/4), "
                 f"v_out = max|{outname}|\n")
        fh.write(f"# outflow guard  = {OUTFLOW_H_GUARD:g} x L{h_label} dropped each "
                 f"{h_label}-end ({h_label} periodic, so not a boundary guard)\n")
        fh.write("#\n")
        fh.write(f"# geometry: sheet along {h_label}, normal along y. "
                 f"INFLOW = y, OUTFLOW = {h_label}.\n")
        fh.write(f"# v_in_CS*   mean inflowing |Vy| in the upstream band\n")
        fh.write(f"# v_out_CS*  max |{outname}| in the outflow band\n")
        if is_flow:
            fh.write("# ratio_CS*  v_in/v_out  -- FLOW RATIO, NOT a reconnection rate\n")
        else:
            fh.write("# vrec_CS*   = v_in/v_out  (dimensionless; NO time deriv, NO vA)\n")
        fh.write("# nin_CS*    number of inflowing samples averaged\n")
        fh.write("#\n")
        rcol = "ratio" if is_flow else "vrec"
        fh.write("# {:>8s} {:>11s} {:>6s} {:>6s} {:>14s} {:>14s} {:>14s} {:>8s} "
                 "{:>14s} {:>14s} {:>14s} {:>8s}\n".format(
                     "cycle", "time", "y_cs1", "y_cs2",
                     "v_in_CS1", "v_out_CS1", f"{rcol}_CS1", "nin1",
                     "v_in_CS2", "v_out_CS2", f"{rcol}_CS2", "nin2"))
        for r in recs:
            fh.write("  {:>8d} {:>11.4f} {:>6d} {:>6d} {:>14.6e} {:>14.6e} {:>14.6e} "
                     "{:>8d} {:>14.6e} {:>14.6e} {:>14.6e} {:>8d}\n".format(
                         r["cycle"], r["t"], r["y1"], r["y2"],
                         r["vin1"], r["vout1"], r["ratio1"], r["nin1"],
                         r["vin2"], r["vout2"], r["ratio2"], r["nin2"]))
    return dict(t=t, r1=r1, r2=r2)


def write_xo_pair_rates(recs, path):
    with open(path, "w") as fh:
        fh.write("# Reconnection rate for individual X-O pairs (2D XY plane)\n")
        fh.write(f"# dir            = {args.dir_data}\n")
        fh.write(f"# sigma_i={args.sigma}  vA/c={vA:.8f}  B0={B0:.8e}  norm={norm:.8e}\n")
        fh.write("# IDs persist by nearest-position matching between dumps.\n")
        fh.write("# Each X paired with BOTH neighboring O points along periodic x.\n")
        fh.write("# dpsi = psi_X - psi_O signed; R = (1/(B0 vA)) d(dpsi)/dt.\n#\n")
        fh.write("# {:>8s} {:>10s} {:>6s} {:>8s} {:>5s} {:>5s} {:>10s} {:>10s} "
                 "{:>10s} {:>10s} {:>15s} {:>15s} {:>15s} {:>15s}\n".format(
                     "cycle", "time", "sheet", "pair_id", "X_id", "O_id",
                     "x_X", "y_X", "x_O", "y_O", "psi_X", "psi_O", "dpsi", "R"))
        for r in sorted(recs, key=lambda q: (q["cycle"], q["sheet"], q["pair_id"])):
            rate = r["ddpsi_dt"] / norm if np.isfinite(r["ddpsi_dt"]) else np.nan
            fh.write("  {:>8d} {:>10.4f} {:>6d} {:>8s} {:>5d} {:>5d} {:>10.4f} "
                     "{:>10.4f} {:>10.4f} {:>10.4f} {:>15.7e} {:>15.7e} {:>15.7e} "
                     "{:>15.7e}\n".format(
                         r["cycle"], r["t"], r["sheet"], r["pair_id"], r["x_id"],
                         r["o_id"], r["x_x"], r["x_y"], r["o_x"], r["o_y"],
                         r["psi_x"], r["psi_o"], r["dpsi"], rate))
    return path


def write_xo(recs, path):
    recs = [r for r in recs if r.get("xo")]
    if not recs:
        return None
    with open(path, "w") as fh:
        fh.write("# X/O identification check (2D XY plane)\n")
        fh.write(f"# dir = {args.dir_data}\n#\n")
        fh.write(f"# by_* : |By| at extremum / mean|By| on the line. Floor ~ "
                 f"2*pi/nxc = {2*np.pi/nxc:.4f}. Values near 1 = failure.\n")
        fh.write("# x_*  : extremum position (code units)\n")
        fh.write("# ez_* : Ez at each point; their DIFFERENCE is the rate.\n#\n")
        fh.write("# {:>8s} {:>10s} {:>11s} {:>11s} {:>10s} {:>10s} {:>12s} {:>12s}"
                 " {:>11s} {:>11s} {:>10s} {:>10s} {:>12s} {:>12s}\n".format(
                     "cycle","time","by_max1","by_min1","x_max1","x_min1",
                     "ez_max1","ez_min1","by_max2","by_min2","x_max2","x_min2",
                     "ez_max2","ez_min2"))
        for r in recs:
            x = r["xo"]; g = lambda k: x.get(k, np.nan)
            fh.write("  {:>8d} {:>10.3f} {:>11.3e} {:>11.3e} {:>10.3f} {:>10.3f}"
                     " {:>12.4e} {:>12.4e} {:>11.3e} {:>11.3e} {:>10.3f}"
                     " {:>10.3f} {:>12.4e} {:>12.4e}\n".format(
                         r["cycle"], r["t"], g("bymax1"), g("bymin1"),
                         g("xmax1"), g("xmin1"), g("ezmax1"), g("ezmin1"),
                         g("bymax2"), g("bymin2"), g("xmax2"), g("xmin2"),
                         g("ezmax2"), g("ezmin2")))
    return path


###! ============================================================
###! Emit outputs
###! ============================================================

if rank == 0:
    os.makedirs(args.outdir, exist_ok=True)

    if is_xy:
        if len(xy_records) < 2:
            raise RuntimeError("Need at least 2 valid XY dumps to differentiate "
                               "in time.")
        norm = B0 * vA

        if xo_pair_records:
            fxp = os.path.join(args.outdir, "reconnection_rate_xo_pairs.txt")
            write_xo_pair_rates(xo_pair_records, fxp)
            print(f"Wrote {fxp}", flush=True)

        f0 = os.path.join(args.outdir, "R_rate_dAz_dt.txt")
        summ = write_flux_table(xy_records, f0,
                                "single XY plane (the only plane; no z-average)")
        print(f"Wrote {f0}", flush=True)
        if args.plot:
            pf = make_flux_plot(summ, os.path.join(args.outdir,
                                "R_rate_dAz_dt.png"), "2D XY plane")
            if pf:
                print(f"Wrote {pf}", flush=True)

        if do_paper and len(paper_records) >= 1:
            fvr = os.path.join(args.outdir, "R_rate_vout_vin.txt")
            vsum = write_paper_table(paper_records, fvr, is_flow=False)
            print(f"Wrote {fvr}", flush=True)
            if args.plot and len(paper_records) >= 2:
                pf = make_paper_plot(vsum, os.path.join(args.outdir, "R_rate_vout_vin.png"),
                                     is_flow=False)
                if pf:
                    print(f"Wrote {pf}", flush=True)

        if args.check_xo:
            fx = write_xo(xy_records, os.path.join(args.outdir, "xo_check.txt"))
            if fx:
                print(f"Wrote {fx}", flush=True)

    else:
        ###! YZ: ONLY the flow diagnostic, and only if we sampled it.
        if not do_paper:
            print("YZ: nothing written (flow diagnostic needs per-species "
                  "moments, or --no-vrec-paper was given).", flush=True)
        elif len(paper_records) < 1:
            print("YZ: no valid cycles produced a flow sample, nothing written.",
                  flush=True)
        else:
            ffl = os.path.join(args.outdir, "flow_diagnostic_YZ.txt")
            fsum = write_paper_table(paper_records, ffl, is_flow=True)
            print(f"Wrote {ffl}  (FLOW DIAGNOSTIC -- not a reconnection rate)",
                  flush=True)
            if args.plot and len(paper_records) >= 2:
                pf = make_paper_plot(fsum, os.path.join(args.outdir,
                                     "flow_diagnostic_YZ.png"), is_flow=True)
                if pf:
                    print(f"Wrote {pf}", flush=True)

    print(f"\nElapsed: {datetime.now() - t_wall}", flush=True)