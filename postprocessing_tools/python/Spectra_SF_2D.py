"""
    srun python3 -u "$SCRIPT" "$DATA_DIR"       \
        $xmin $xmax $ymin $ymax $zmin $zmax     \
        --nxc "$nxc" --nyc "$nyc" --nzc "$nzc"  \
        --cycle-start 0 --cycle-end 20000 --cycle-step 100 --time-step 20 \
        --outdir "$OUT_DIR"

    Cache written: <outdir>/spectra_sf_cache.npz  (override with --cache-name).
"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import glob
import argparse

import h5py
import numpy as np
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

###! ------------------------------------------------------------------
COMPONENTS = ["Bx", "By", "Bz"]        ###! magnetic vector components (dB)
CS_SLABS   = [(0.225, 0.275), (0.725, 0.775)]
SF_ORDERS  = (4,)                      ###! raw moment <|df|^4> cached (even order)
SF_MIN_LAG_CELLS = 1

###! ------------------------------------------------------------------
###! Species groupings (0-indexed), matching rho_CS.py / B_J.py conventions.
###! Each group is SUMMED in real space (sum-then-FFT) to form the group fluid
###! moment before the fluctuation and spectrum are taken.
ELECTRONS = [0, 2]
PROTONS   = [1, 3]

###! Scalar per-species moment datasets we spectrally analyse.
SCALAR_QTYS = ["Jz", "rho"]           ###! read moments/species_{s}/{Jz,rho}/...

###! Which physical axis each 2D plane collapses (matches B_J.py coll_axis_of).
###!   XY collapses z ; YZ collapses x. (ZX is not offered here: this diagnostic
###!   needs y resolved, and ZX collapses y.)
COLL_AXIS_OF = {"XY": 2, "YZ": 0}

comm.Barrier()


def proc_id_from_filename(fp):
    base = os.path.basename(fp)
    return int(base.replace("proc", "").replace(".hdf", ""))


def mapping_candidates(XLEN, YLEN, ZLEN):
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
    ###! Score each candidate proc -> (i,j,k) by grid occupancy; the correct one
    ###! covers every tile exactly once (score 0). NOTE: with a collapsed axis
    ###! (LEN = 1) several candidates can tie, since permuting a length-1 axis
    ###! changes nothing. On a tie this returns the first best; pass --mapping to
    ###! force one if the auto choice is ever wrong for your file ordering.
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


def global_shape_shared(tile_shape, XLEN, YLEN, ZLEN):
    ###! Shared-boundary assembly: global nodes = LEN*(n_tile - 1) + 1. On a
    ###! collapsed axis (LEN = 1, n_tile = 2) this gives 2 nodes, of which we
    ###! sample the lower one.
    nx, ny, nz = tile_shape
    return (XLEN * (nx - 1) + 1, YLEN * (ny - 1) + 1, ZLEN * (nz - 1) + 1)


###! ------------------------------------------------------------------
###! Slab assembler for a SINGLE resolved 2D plane, restricted to a y-slab.
###!
###! Assembles every field in `datasets` on the resolved plane (H, y), where the
###! collapsed axis is sampled at its single global node `coll_index` (= 0), and
###! y is restricted to [jlo, jhi). Reduces to rank 0.
###!
###!   plane XY: H = x  (collapse z)  -> returns arrays (Gx, ny_slab)
###!   plane YZ: H = z  (collapse x)  -> returns arrays (Gz, ny_slab)
###!
###! Duplicate shared-boundary planes are averaged via a per-node count, exactly
###! as the 3D assembler did, so the HDF5 is opened ONCE per tile. Per-species
###! scalars are kept as separate keys and summed by the caller after assembly.
###! ------------------------------------------------------------------

def assemble_plane_slab(cycle_name, local_files, rank_to_ijk, tile_shape,
                        G_shape, jlo, jhi, datasets, plane, coll_index):
    """COLLECTIVE. Returns {key: (GH, ny_slab)} on rank 0, else None.
    GH is Gx for XY, Gz for YZ."""
    Gx, Gy, Gz = G_shape
    nx_t, ny_t, nz_t = tile_shape
    nx_c, ny_c, nz_c = nx_t - 1, ny_t - 1, nz_t - 1
    ny_slab = jhi - jlo

    ###! resolved horizontal (H) global size and which raw axis it maps to
    if plane == "XY":
        GH = Gx
        h_c = nx_c
    else:                                   ###! YZ
        GH = Gz
        h_c = nz_c

    keys = list(datasets.keys())
    local = {k: np.zeros((GH, ny_slab), dtype=np.float64) for k in keys}
    cnt = np.zeros((GH, ny_slab), dtype=np.float64)

    for fp in local_files:
        i, j, k = rank_to_ijk(proc_id_from_filename(fp))
        xs = 0 if i == 0 else 1
        ys = 0 if j == 0 else 1
        zs = 0 if k == 0 else 1
        gx0, gy0, gz0 = i * nx_c + xs, j * ny_c + ys, k * nz_c + zs
        nxu, nyu, nzu = nx_t - xs, ny_t - ys, nz_t - zs

        ###! y-slab overlap for this tile (identical logic to the 3D assembler)
        a = max(gy0, jlo)
        b = min(gy0 + nyu - 1, jhi - 1)
        if a > b:
            continue
        js = a - gy0 + ys                   ###! local y start (skip shared plane)
        je = b - gy0 + ys + 1
        oy = a - jlo                        ###! output y offset into the slab
        ny_read = je - js

        ###! ---- collapsed-axis ownership: does this tile hold coll_index? ----
        ###! coll_index is a GLOBAL node index on the collapsed axis (always 0
        ###! here). The tile owns it iff its global node range covers it. The
        ###! local index into the tile is coll_index - global_offset, where the
        ###! offset uses the UNCROPPED tile origin (i*n_c), because coll_index is
        ###! measured from global node 0.
        if plane == "XY":                   ###! collapse z
            c_o = k * nz_c                  ###! this tile's global z origin (uncropped)
            c_lo = gz0                      ###! cropped lower owned node
            c_hi = k * nz_c + nz_t - 1      ###! uncropped upper owned node
            h_gx0, h_nu = gx0, nxu          ###! H = x
        else:                               ###! YZ, collapse x
            c_o = i * nx_c
            c_lo = gx0
            c_hi = i * nx_c + nx_t - 1
            h_gx0, h_nu = gz0, nzu          ###! H = z

        if not (c_lo <= coll_index <= c_hi):
            continue
        c_local = coll_index - c_o          ###! local index on the collapsed axis

        with h5py.File(fp, "r") as f:
            for key, tmpl in datasets.items():
                path = tmpl.format(cycle=cycle_name)
                if path not in f:
                    raise KeyError(
                        f"Missing dataset {path} in {os.path.basename(fp)}")
                d = f[path]
                if plane == "XY":
                    ###! keep (x, y) at fixed z = c_local ; crop shared x,y planes
                    blk = np.asarray(d[xs:, js:je, c_local], dtype=np.float64)   ###! [x, y]
                else:
                    ###! keep (z, y) at fixed x = c_local ; result wanted [z, y]
                    slab_yz = np.asarray(d[c_local, js:je, zs:], dtype=np.float64)  ###! [y, z]
                    blk = slab_yz.T                                              ###! [z, y]
                local[key][h_gx0:h_gx0+h_nu, oy:oy+ny_read] += blk
        cnt[h_gx0:h_gx0+h_nu, oy:oy+ny_read] += 1.0

    for key in keys:
        comm.Allreduce(MPI.IN_PLACE, local[key], op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, cnt, op=MPI.SUM)

    if rank != 0:
        return None
    ok = cnt > 0
    for key in keys:
        local[key][ok] /= cnt[ok]
    if not ok.all():
        print(f"  WARNING: {int((~ok).sum())} slab nodes never written at "
              f"{cycle_name}", flush=True)
    return local


def fluctuation_vec(slab):
    """dB_i = B_i - <B_i>_x(y): subtract the x-average PER y-row (axis 0), for
    each magnetic component. Removes the Harris tanh profile and any y-dependent
    mean before the FFT. `slab` is a dict of 2D arrays (GH, ny_slab)."""
    return {c: slab[c] - slab[c].mean(axis=0, keepdims=True)
            for c in COMPONENTS}


def fluctuation_scalar(arr):
    """df = f - <f>_x(y): x-average subtracted per y-row for a single 2D scalar
    slab (GH, ny_slab)."""
    return arr - arr.mean(axis=0, keepdims=True)


def build_axis_binning(n, L):
    ###! 1D power-of-2pi/L binning, identical to the 3D script. Returns the bin
    ###! index per FFT mode, bin count, bin-centre |k|, and a validity mask
    ###! (positive k, within Nyquist).
    k0 = 2.0 * np.pi / L
    ka = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    kabs = np.abs(ka)
    idx = np.floor(kabs / k0 + 0.5).astype(np.int64)
    n_bins = int(idx.max()) + 1
    counts = np.bincount(idx, minlength=n_bins)
    k_sum = np.bincount(idx, weights=kabs, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        kc = np.where(counts > 0, k_sum / np.maximum(counts, 1), np.nan)
    kny = np.pi * n / L
    valid = (counts > 0) & (kc <= kny) & (np.arange(n_bins) > 0)
    return idx, n_bins, kc, valid


def power_2d_vector(field):
    """0.5 * sum_i |F_i(kH, ky)|^2 for the magnetic VECTOR. `field` is a dict of
    3 real 2D slabs (H_fft, ny_fft). FFT over BOTH plane axes. Returns the 2D
    power (nH, nky). (3D convention: components summed, factor 1/2.)"""
    nH, ny = field[COMPONENTS[0]].shape
    norm = 1.0 / (nH * ny)
    power = np.zeros((nH, ny), dtype=np.float64)
    for comp in COMPONENTS:
        Fhat = np.fft.fft2(field[comp]) * norm
        power += 0.5 * (np.abs(Fhat) ** 2)
    return power


def power_2d_scalar(arr):
    """|F(kH, ky)|^2 for a SCALAR 2D slab (H_fft, ny_fft). FFT over both plane
    axes. No 1/2, no component sum. Returns (nH, nky)."""
    nH, ny = arr.shape
    norm = 1.0 / (nH * ny)
    Fhat = np.fft.fft2(arr) * norm
    return np.abs(Fhat) ** 2


def reduce_axis(power_2d, keep, idx, n_bins):
    """Marginal 1D spectrum. keep 'H' -> keep kH (sum over ky, axis 1); keep 'y'
    -> keep ky (sum over kH, axis 0). Then bin onto the |k| grid."""
    per_axis = power_2d.sum(axis=1) if keep == "H" else power_2d.sum(axis=0)
    return np.bincount(idx, weights=per_axis, minlength=n_bins)


def build_sf_lags(n, max_frac):
    hi = max(SF_MIN_LAG_CELLS + 1, int(np.floor(max_frac * n)))
    lags = np.unique(np.round(
        np.geomspace(SF_MIN_LAG_CELLS, hi, num=40)).astype(int))
    return lags[(lags >= SF_MIN_LAG_CELLS) & (lags <= hi)]


def sf_moments(field, axis, lags, orders):
    """Raw structure-function moments <|df|^n> along one axis of the 2D field
    for each even n in `orders`. `field` is the magnetic VECTOR dict; the SF is
    on the vector increment magnitude. axis: 0 = H (x or z), 1 = y.

    ###! Fast path (exact for even n): work in d2 = |df|^2 = sum_i df_i^2 and
    ###! form |df|^n = (d2)^(n/2) by integer powers; increments cast to float32
    ###! (memory-bandwidth bound), reduction accumulated in float64. Odd orders
    ###! would need sqrt(d2) and are rejected. Identical to the 3D routine but
    ###! over a 2D slab.
    """
    for n in orders:
        if n % 2 != 0:
            raise ValueError(f"sf_moments fast path assumes even orders; got n={n}")

    n_axis = field[COMPONENTS[0]].shape[axis]
    out = {n: np.full(lags.shape, np.nan, dtype=np.float64) for n in orders}
    half = {n: n // 2 for n in orders}
    max_half = max(half.values())

    for li, l in enumerate(lags):
        if l >= n_axis:
            continue
        sl_hi = [slice(None)] * 2
        sl_lo = [slice(None)] * 2
        sl_hi[axis] = slice(l, n_axis)
        sl_lo[axis] = slice(0, n_axis - l)

        d2 = None
        for comp in COMPONENTS:
            d = (field[comp][tuple(sl_hi)] -
                 field[comp][tuple(sl_lo)]).astype(np.float32)
            d2 = d * d if d2 is None else d2 + d * d

        p = d2
        power_cache = {1: d2}
        for e in range(2, max_half + 1):
            p = p * d2
            power_cache[e] = p
        for n in orders:
            out[n][li] = np.mean(power_cache[half[n]], dtype=np.float64)

    return out


###! ============================================================
###! Arguments
###! ============================================================

parser = argparse.ArgumentParser(
    description="MPI compute stage (2D XY/YZ): per-CS-averaged in-plane spectra "
                "E(kH) and E(ky) for dB (summed vector), dJz and drho "
                "(per-species-summed scalars), plus SF moments <|df|^4> for B "
                "and dB along the two resolved axes, to a .npz cache. FFT is "
                "over the FULL resolved plane restricted to the chosen y-slab; "
                "df = f - <f>_x(y) removes the Harris profile. No k_perp.")
parser.add_argument("dir_data", type=str)
parser.add_argument("xmin", type=float); parser.add_argument("xmax", type=float)
parser.add_argument("ymin", type=float); parser.add_argument("ymax", type=float)
parser.add_argument("zmin", type=float); parser.add_argument("zmax", type=float)
parser.add_argument("--nxc", type=int, required=True)
parser.add_argument("--nyc", type=int, required=True)
parser.add_argument("--nzc", type=int, required=True)
parser.add_argument("--plane", type=str, required=True, choices=["XY", "YZ"],
                    help="Resolved plane. XY collapses z (nzc must be 1); "
                         "YZ collapses x (nxc must be 1). y stays resolved in both.")
parser.add_argument("--cycle-start", type=int, default=0)
parser.add_argument("--cycle-end", type=int, default=5000)
parser.add_argument("--cycle-step", type=int, default=100)
###! physical-time label per OUTPUT FRAME: time = (cycle / cycle_step) * time_step
parser.add_argument("--time-step", type=float, default=100.0)
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--cache-name", type=str, default="spectra_sf_cache.npz")
parser.add_argument("--mapping", type=str, default="auto",
                    choices=["auto", "A", "B", "C", "D", "E", "F"])
args = parser.parse_args()

nxc, nyc, nzc = args.nxc, args.nyc, args.nzc
plane = args.plane
Lx = args.xmax - args.xmin
Ly = args.ymax - args.ymin
Lz = args.zmax - args.zmin
outdir = args.dir_data if args.outdir is None else args.outdir
if rank == 0:
    os.makedirs(outdir, exist_ok=True)

###! ---- resolve/validate the 2D mode (collapsed axis must have nc == 1) ----
###! Mirrors B_J.py's mode check: the named plane's collapsed axis must be the
###! nc == 1 one, and no OTHER axis may be collapsed. y must be resolved.
ncs = (nxc, nyc, nzc)
ca = COLL_AXIS_OF[plane]
mode_error = None
if ncs[ca] != 1:
    mode_error = (f"--plane {plane} collapses {'xyz'[ca]}, but "
                  f"n{'xyz'[ca]}c = {ncs[ca]} (expected 1 for a 2D-spatial run).")
else:
    for a in range(3):
        if a != ca and ncs[a] == 1:
            mode_error = (f"--plane {plane} but n{'xyz'[a]}c = 1 as well: more "
                          f"than one axis is collapsed, not a 2D plane.")
            break
    if ncs[1] == 1:
        mode_error = "y is collapsed (nyc == 1); this diagnostic needs y resolved."
if mode_error is not None:
    if rank == 0:
        print(f"MODE ERROR: {mode_error}", flush=True)
    comm.Barrier(); raise SystemExit(1)

###! The resolved horizontal axis (H): x for XY, z for YZ. Its physical length
###! and cell count set the kH binning.
if plane == "XY":
    LH = Lx
    nHc = nxc
    h_label = "x"
else:                                   ###! YZ
    LH = Lz
    nHc = nzc
    h_label = "z"

###! ---------------- setup ----------------
setup_error = None
if rank == 0:
    try:
        all_files = sorted(glob.glob(os.path.join(args.dir_data, "proc*.hdf")))
        if not all_files:
            raise RuntimeError(f"No proc*.hdf files found in {args.dir_data}")
        with h5py.File(all_files[0], "r") as f:
            tile_shape = tuple(f[f"fields/Bx/cycle_{args.cycle_start}"].shape)

            ###! ---- probe which per-species scalar moments are available ----
            ###! (identical policy to the 3D script: a scalar GROUP is usable
            ###! only if ALL its member species are present for that quantity;
            ###! partial groups are dropped, not partially summed.)
            c0 = f"cycle_{args.cycle_start}"

            def sp_present(s, qty):
                p = f"moments/species_{s}/{qty}/{c0}"
                if p not in f:
                    return False
                ms = tuple(f[p].shape)
                if ms != tile_shape:
                    raise RuntimeError(
                        f"Moment '{qty}' species {s} tile shape {ms} != Bx "
                        f"tile shape {tile_shape}; assembly indexing would be "
                        f"wrong. Refusing to assemble a corrupt field.")
                return True

            groups = {"e": ELECTRONS, "p": PROTONS}
            avail = {}
            missing_groups = []
            for qty in SCALAR_QTYS:
                for gt, members in groups.items():
                    if all(sp_present(s, qty) for s in members):
                        avail[(qty, gt)] = list(members)
                    else:
                        have = [s for s in members if sp_present(s, qty)]
                        missing_groups.append(
                            f"{qty}/{gt}(need {members}, have {have})")

            qty_prefix = {"Jz": "dJz", "rho": "dr"}
            spec_tags = ["dB"]
            tag_source = {}
            for (qty, gt), members in avail.items():
                tag = f"{qty_prefix[qty]}{gt}"
                spec_tags.append(tag)
                tag_source[tag] = (qty, members)

            print(f"Scalar moments  : "
                  f"{sorted(tag_source) if tag_source else '(none -- dB only)'}",
                  flush=True)
            if missing_groups:
                print(f"  SKIPPED groups: {'; '.join(missing_groups)}",
                      flush=True)

        nx_t, ny_t, nz_t = tile_shape
        ###! derive decomposition; a collapsed axis has nc = 1, n_tile = 2 -> LEN 1
        for lbl, n_c, n_t in (("x", nxc, nx_t), ("y", nyc, ny_t), ("z", nzc, nz_t)):
            if n_t < 2 or n_c % (n_t - 1) != 0:
                raise RuntimeError(
                    f"Cannot derive the {lbl} decomposition: {n_c} cells do not "
                    f"divide into tiles of {n_t - 1} cells (tile {tile_shape}).")
        XLEN, YLEN, ZLEN = nxc // (nx_t-1), nyc // (ny_t-1), nzc // (nz_t-1)
        if XLEN * YLEN * ZLEN != len(all_files):
            raise RuntimeError(f"decomposition {XLEN}x{YLEN}x{ZLEN} != "
                               f"{len(all_files)} files")
        ###! collapsed axis must be a single tile (LEN == 1) or the "sample the
        ###! lower node" assumption is wrong.
        coll_len = (ZLEN if plane == "XY" else XLEN)
        if coll_len != 1:
            raise RuntimeError(
                f"Collapsed axis has {coll_len} tiles (expected 1 for plane "
                f"{plane}). A 2D run must not decompose the collapsed axis.")
        map_name = (choose_mapping(all_files, XLEN, YLEN, ZLEN)
                    if args.mapping == "auto" else args.mapping)
    except Exception as exc:
        setup_error = f"{type(exc).__name__}: {exc}"
        all_files = map_name = tile_shape = None
        XLEN = YLEN = ZLEN = None
        spec_tags = tag_source = None
else:
    all_files = map_name = tile_shape = None
    XLEN = YLEN = ZLEN = None
    spec_tags = tag_source = None

setup_error = comm.bcast(setup_error, root=0)
if setup_error is not None:
    if rank == 0:
        print(f"SETUP FAILED: {setup_error}", flush=True)
    comm.Barrier(); raise SystemExit(1)

all_files  = comm.bcast(all_files, root=0)
map_name   = comm.bcast(map_name, root=0)
tile_shape = comm.bcast(tile_shape, root=0)
XLEN = comm.bcast(XLEN, root=0)
YLEN = comm.bcast(YLEN, root=0)
ZLEN = comm.bcast(ZLEN, root=0)
spec_tags = comm.bcast(spec_tags, root=0)
tag_source = comm.bcast(tag_source, root=0)

rank_to_ijk = mapping_candidates(XLEN, YLEN, ZLEN)[map_name]
local_files = all_files[rank::size]
G_shape = global_shape_shared(tile_shape, XLEN, YLEN, ZLEN)
Gx, Gy, Gz = G_shape

###! collapsed axis is sampled at its lower global node (matches B_J.py's
###! coll_index = 0). With LEN == 1 there are exactly two global nodes (0 and 1,
###! the periodic image); node 0 is the physical plane.
coll_index = 0

###! datasets: the 3 B components + only the per-species Jz/rho found available.
DATASETS = {c: f"fields/{c}/{{cycle}}" for c in COMPONENTS}
needed_species_qty = set()
for tag, (qty, members) in tag_source.items():
    for s in members:
        needed_species_qty.add((s, qty))
for (s, qty) in sorted(needed_species_qty):
    DATASETS[f"{qty}_s{s}"] = f"moments/species_{s}/{qty}/{{cycle}}"

requested = list(range(args.cycle_start, args.cycle_end + 1, args.cycle_step))

probe_error = None
if rank == 0:
    try:
        with h5py.File(all_files[0], "r") as f:
            cycle_names, missing = [], []
            for c in requested:
                nm = f"cycle_{c}"
                (cycle_names if f"fields/Bx/{nm}" in f else missing).append(nm)
            if not cycle_names:
                raise RuntimeError("None of the requested cycles are present.")
            if missing:
                print(f"WARNING: {len(missing)} cycle(s) absent: "
                      f"{missing[0]} ... {missing[-1]}", flush=True)
    except Exception as exc:
        probe_error = f"{type(exc).__name__}: {exc}"
        cycle_names = None
else:
    cycle_names = None
probe_error = comm.bcast(probe_error, root=0)
if probe_error is not None:
    if rank == 0:
        print(f"PROBE FAILED: {probe_error}", flush=True)
    comm.Barrier(); raise SystemExit(1)
cycle_names = comm.bcast(cycle_names, root=0)

###! y-slab node ranges for the two current sheets (unchanged from 3D: fractions
###! of nyc, clamped to the global y range).
slab_ranges = []
for (f_lo, f_hi) in CS_SLABS:
    jlo = max(0, int(np.floor(f_lo * nyc)))
    jhi = min(Gy, int(np.ceil(f_hi * nyc)))
    slab_ranges.append((jlo, jhi))

###! FFT grid sizes. H (resolved horizontal) is periodic -> drop the duplicated
###! top node: nH_fft = GH - 1 = nHc. y is NOT periodic-wrapped within a slab and
###! is not a full-box axis, so the slab is FFT'd at its full node count. (A
###! non-periodic y-slab FFT implies an implicit windowing/periodicity assumption
###! in ky; the ky spectrum is therefore only meaningful for k*L_slab >> 1. This
###! is flagged in the cache metadata as ky_slab_caveat.)
if plane == "XY":
    nH_fft = Gx - 1
else:
    nH_fft = Gz - 1

if rank == 0:
    ###! kH binning on the resolved horizontal axis (full box, periodic).
    idx_H, nbH, kcH, vH = build_axis_binning(nH_fft, LH)
    ###! ky binning uses the SLAB physical length, not Ly: L_yslab = ny_slab*dy.
    ###! Built per-slab inside the loop because ny_slab can differ if the two
    ###! CS fractions round to different node counts; here we assume they match
    ###! (they do for symmetric fractions) and build from slab 0.
    dy = Ly / nyc
    ny_slab0 = slab_ranges[0][1] - slab_ranges[0][0]
    Ly_slab = ny_slab0 * dy
    idx_y, nby, kcy, vy = build_axis_binning(ny_slab0, Ly_slab)

    lags_H = build_sf_lags(nH_fft, max_frac=0.5)
    lags_y = build_sf_lags(ny_slab0, max_frac=0.9)

    ###! guard: the two CS slabs must have equal node counts, or the ky binning
    ###! and the SF-y lags (built from slab 0) would not apply to slab 1.
    ny_slab1 = slab_ranges[1][1] - slab_ranges[1][0]
    if ny_slab1 != ny_slab0:
        raise SystemExit(
            f"The two CS y-slabs have different node counts "
            f"({ny_slab0} vs {ny_slab1}); ky binning and SF-y lags assume they "
            f"match. Adjust CS_SLABS to symmetric fractions.")

    ###! store keys: Ex_/Ez_ for the resolved horizontal, Ey_ for ky. dB always
    ###! plus available scalar groups. SF on B and dB along H and y only.
    h_key = "Ex" if plane == "XY" else "Ez"
    def new_store():
        s = {"SF_B": {n: {"H": [], "y": []} for n in SF_ORDERS},
             "SF_dB": {n: {"H": [], "y": []} for n in SF_ORDERS}}
        for tag in spec_tags:
            s[f"{h_key}_{tag}"] = []
            s[f"Ey_{tag}"] = []
        return s
    store = {cs: new_store() for cs in range(len(CS_SLABS))}

###! ---------------- main loop ----------------
for cyc in cycle_names:
    for cs, (jlo, jhi) in enumerate(slab_ranges):
        slab = assemble_plane_slab(cyc, local_files, rank_to_ijk, tile_shape,
                                   G_shape, jlo, jhi, DATASETS, plane, coll_index)
        if rank != 0:
            continue

        ###! ---- magnetic vector: crop the shared duplicate H plane, form dB ----
        ###! crop H to nH_fft (drop the periodic top node); keep full y-slab.
        B = {c: slab[c][:nH_fft, :] for c in COMPONENTS}
        dB = fluctuation_vec(B)             ###! df = f - <f>_x(y) per y-row

        p_dB = power_2d_vector(dB)
        store[cs][f"{h_key}_dB"].append(reduce_axis(p_dB, "H", idx_H, nbH))
        store[cs][f"Ey_dB"].append(reduce_axis(p_dB, "y", idx_y, nby))

        ###! ---- per-species SUM-THEN-FFT for each available scalar group ----
        for tag, (qty, members) in tag_source.items():
            acc = None
            for s in members:
                a = slab[f"{qty}_s{s}"][:nH_fft, :]
                acc = a.copy() if acc is None else acc + a
            df = fluctuation_scalar(acc)   ###! df = f - <f>_x(y) per y-row
            P = power_2d_scalar(df)
            store[cs][f"{h_key}_{tag}"].append(reduce_axis(P, "H", idx_H, nbH))
            store[cs][f"Ey_{tag}"].append(reduce_axis(P, "y", idx_y, nby))

        ###! ---- SF moments on B and dB along H (axis 0) and y (axis 1) ----
        for tag, fld in (("SF_B", B), ("SF_dB", dB)):
            mH = sf_moments(fld, 0, lags_H, SF_ORDERS)
            my = sf_moments(fld, 1, lags_y, SF_ORDERS)
            for n in SF_ORDERS:
                store[cs][tag][n]["H"].append(mH[n])
                store[cs][tag][n]["y"].append(my[n])

    if rank == 0:
        print(f"{cyc} done", flush=True)

###! ---------------- average sheets + write cache ----------------
if rank == 0:
    def avg2(key):
        return 0.5 * (np.array(store[0][key]) + np.array(store[1][key]))

    def avg2_sf(tag, n, ax):
        a = np.array(store[0][tag][n][ax])
        b = np.array(store[1][tag][n][ax])
        return 0.5 * (a + b)

    cycles = np.array([int(c.replace("cycle_", "")) for c in cycle_names])
    times = cycles / args.cycle_step * args.time_step

    save = dict(
        # axes / masks: kc_H (+valid_H) is the resolved horizontal, kc_y the slab ky
        kc_H=kcH, valid_H=vH, kc_y=kcy, valid_y=vy,
        lags_H=lags_H, lags_y=lags_y,
        dx=Lx / nxc, dy=Ly / nyc, dz=Lz / nzc if nzc > 0 else 0.0,
        cycles=cycles, times=times,
        # meta
        plane=np.array([plane], dtype=object),
        h_label=np.array([h_label], dtype=object),      ###! 'x' (XY) or 'z' (YZ)
        h_key=np.array([h_key], dtype=object),           ###! 'Ex' or 'Ez'
        LH=LH, Ly_slab=ny_slab0 * (Ly / nyc),
        sf_orders=np.array(SF_ORDERS),
        cs_slabs=np.array(CS_SLABS),
        grid=np.array([nxc, nyc, nzc]),
        box=np.array([Lx, Ly, Lz]),
        mapping=np.array([map_name], dtype=object),
        electrons=np.array(ELECTRONS),
        protons=np.array(PROTONS),
        spec_tags=np.array(spec_tags, dtype=object),
        ###! honest caveat carried in the cache: ky is an FFT over a NON-periodic
        ###! thin y-slab, so ky modes are only meaningful for k*Ly_slab >> 1.
        ky_slab_caveat=np.array(
            ["ky FFT is over a non-periodic y-slab; trust only k*Ly_slab >> 1"],
            dtype=object),
    )

    ###! spectra (CS-averaged): horizontal (Ex_ or Ez_) and Ey_ per tag
    for tag in spec_tags:
        save[f"{h_key}_{tag}"] = avg2(f"{h_key}_{tag}")
        save[f"Ey_{tag}"] = avg2(f"Ey_{tag}")

    ###! SF moments: one array per (field, order, direction) for H and y
    for tag in ("SF_B", "SF_dB"):
        for n in SF_ORDERS:
            save[f"{tag}{n}_H"] = avg2_sf(tag, n, "H")
            save[f"{tag}{n}_y"] = avg2_sf(tag, n, "y")

    cache_path = os.path.join(outdir, args.cache_name)
    np.savez_compressed(cache_path, **save)
    print(f"Wrote cache: {cache_path}", flush=True)
    print(f"  plane={plane}  H={h_label}  spectra tags={spec_tags}", flush=True)
    print(f"  reductions: {h_key}_ (k{h_label}), Ey_ (ky over the CS slab)",
          flush=True)
    print(f"  fluctuation: df = f - <f>_{h_label}(y) (Harris profile removed)",
          flush=True)
    print(f"  SF orders={SF_ORDERS} on B and dB along {h_label} and y "
          f"(raw moments <|df|^n>, CS-averaged)", flush=True)
    print(f"  k_perp: DROPPED (thin y-slab is not a square grid)", flush=True)
    print(f"  per-species groups: electrons={ELECTRONS} protons={PROTONS} "
          f"(summed before FFT)", flush=True)

comm.Barrier()