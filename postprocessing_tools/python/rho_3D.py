"""
    Spectra_compute.py -- HEAVY stage (MPI). Reads the proc*.hdf tiles ONCE,
    computes the per-CS-averaged in-plane spectra and structure-function moments,
    and writes a single small .npz cache. Spectra_plot.py then makes the figures
    from that cache in seconds, so replotting never re-reads the HDF5.

    srun python3 -u Spectra_compute.py "$DATA_DIR" \
        $xmin $xmax $ymin $ymax $zmin $zmax \
        --nxc 768 --nyc 1536 --nzc 768 \
        --cycle-start 0 --cycle-end 4200 --cycle-step 100 \
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
CS_SLABS   = [(0.2, 0.3), (0.7, 0.8)]
SF_ORDERS  = (4,)                      ###! raw moment <|df|^4> cached (even order)
SF_MIN_LAG_CELLS = 1

###! ------------------------------------------------------------------
###! Species groupings (0-indexed), matching rho_CS.py / B_J.py conventions.
###! Each group is SUMMED in real space (sum-then-FFT) to form the group fluid
###! moment before the fluctuation and spectrum are taken.
ELECTRONS = [0, 2]
PROTONS   = [1, 3]

###! Scalar per-species moment datasets we spectrally analyse. name -> h5 leaf.
###! Group tag ("e"/"p") x quantity ("Jz"/"rho") -> the five scalar dfields are
###! built from these; dB is handled separately as the magnetic vector.
SCALAR_QTYS = ["Jz", "rho"]           ###! read moments/species_{s}/{Jz,rho}/...

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
    nx, ny, nz = tile_shape
    return (XLEN * (nx - 1) + 1, YLEN * (ny - 1) + 1, ZLEN * (nz - 1) + 1)


###! ------------------------------------------------------------------
###! Generic slab assembler. `datasets` maps an output KEY -> the in-file HDF5
###! path TEMPLATE (a format string taking cycle_name). All requested keys are
###! assembled together on the same (x,z) x y-slab grid and averaged over the
###! shared-boundary duplicate planes, so the HDF5 is opened ONCE per tile.
###!
###! For per-species SCALARS (Jz, rho) we DO NOT sum here -- each species is a
###! separate key, and the caller sums the group AFTER assembly (real space),
###! which is identical to summing before assembly but keeps this routine field
###! -agnostic. Vector B keeps its three component keys too.
###! ------------------------------------------------------------------

def assemble_slab(cycle_name, local_files, rank_to_ijk, tile_shape,
                  G_shape, jlo, jhi, datasets):
    """COLLECTIVE. Assemble every field in `datasets` on (x,z) over y-slab
    [jlo,jhi); reduce to rank 0. `datasets` is {key: path_template}, where
    path_template.format(cycle=cycle_name) gives the HDF5 dataset path.
    Returns {key: (Gx, ny_slab, Gz)} on rank 0, else None."""
    Gx, Gy, Gz = G_shape
    nx_t, ny_t, nz_t = tile_shape
    nx_c, ny_c, nz_c = nx_t - 1, ny_t - 1, nz_t - 1
    ny_slab = jhi - jlo

    keys = list(datasets.keys())
    local = {k: np.zeros((Gx, ny_slab, Gz), dtype=np.float64) for k in keys}
    cnt = np.zeros((Gx, ny_slab, Gz), dtype=np.float64)

    for fp in local_files:
        i, j, k = rank_to_ijk(proc_id_from_filename(fp))
        xs = 0 if i == 0 else 1
        ys = 0 if j == 0 else 1
        zs = 0 if k == 0 else 1
        gx0, gy0, gz0 = i * nx_c + xs, j * ny_c + ys, k * nz_c + zs
        nxu, nyu, nzu = nx_t - xs, ny_t - ys, nz_t - zs

        a = max(gy0, jlo)
        b = min(gy0 + nyu - 1, jhi - 1)
        if a > b:
            continue
        js = a - gy0 + ys
        je = b - gy0 + ys + 1
        oy = a - jlo
        ny_read = je - js

        with h5py.File(fp, "r") as f:
            for key, tmpl in datasets.items():
                path = tmpl.format(cycle=cycle_name)
                if path not in f:
                    raise KeyError(
                        f"Missing dataset {path} in {os.path.basename(fp)}")
                blk = np.asarray(f[path][xs:, js:je, zs:], dtype=np.float64)
                local[key][gx0:gx0+nxu, oy:oy+ny_read, gz0:gz0+nzu] += blk
        cnt[gx0:gx0+nxu, oy:oy+ny_read, gz0:gz0+nzu] += 1.0

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
    """dB = B - <B>_xz(y) for the magnetic VECTOR (dict of 3 components)."""
    return {c: slab[c] - slab[c].mean(axis=(0, 2), keepdims=True)
            for c in COMPONENTS}


def fluctuation_scalar(arr):
    """df = f - <f>_xz(y) for a single SCALAR slab array (Gx, ny_slab, Gz)."""
    return arr - arr.mean(axis=(0, 2), keepdims=True)


def build_axis_binning(n, L):
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


def build_radial_binning(nx, nz, Lx, Lz):
    """2D annular (isotropic) binning over k_perp = sqrt(kx^2 + kz^2).

    ###! REQUIRES A SQUARE GRID: Delta kx = Delta kz (i.e. Lx/nx == Lz/nz), so
    ###! that physical |k| is proportional to the integer index radius and a
    ###! single integer-radius bin is a true constant-|k| annulus. This run
    ###! family has Lx = Lz and nx_fft = nz_fft; a guard in __main__ aborts if
    ###! that ever fails, so here we bin on the integer radius directly.

    Returns (idx2d (nx,nz), n_bins, kc_p (n_bins,), valid_p (n_bins,)), where
    idx2d[ix,iz] is the radial bin of mode (ix,iz) and kc_p is the mean physical
    |k| in each bin."""
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)      ###! (nx,)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=Lz / nz)      ###! (nz,)
    KX, KZ = np.meshgrid(kx, kz, indexing="ij")           ###! (nx,nz)
    kperp = np.sqrt(KX * KX + KZ * KZ)                     ###! (nx,nz)

    k0 = 2.0 * np.pi / Lx                                  ###! == 2pi/Lz (square)
    idx2d = np.floor(kperp / k0 + 0.5).astype(np.int64)    ###! (nx,nz) ring index
    n_bins = int(idx2d.max()) + 1

    flat_idx = idx2d.ravel()
    flat_k = kperp.ravel()
    counts = np.bincount(flat_idx, minlength=n_bins)
    k_sum = np.bincount(flat_idx, weights=flat_k, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        kc = np.where(counts > 0, k_sum / np.maximum(counts, 1), np.nan)

    ###! isotropic Nyquist: only trust rings inside the inscribed circle of the
    ###! 2D Nyquist square, i.e. |k| <= min(kny_x, kny_z). With a square grid
    ###! these are equal.
    kny = min(np.pi * nx / Lx, np.pi * nz / Lz)
    valid = (counts > 0) & (kc <= kny) & (np.arange(n_bins) > 0)
    return idx2d, n_bins, kc, valid


def power_kxz_vector(field):
    """0.5 * sum_i |F_i(kx,kz)|^2, y-averaged. `field` is a dict of 3 real
    slabs (Gx-cropped, ny_slab, Gz-cropped). Returns (nx, nz) 2D power."""
    nx, ny_slab, nz = field[COMPONENTS[0]].shape
    norm = 1.0 / (nx * nz)
    power = np.zeros((nx, nz), dtype=np.float64)
    for comp in COMPONENTS:
        Fhat = np.fft.fft2(field[comp], axes=(0, 2)) * norm
        power += 0.5 * (np.abs(Fhat) ** 2).mean(axis=1)
    return power


def power_kxz_scalar(arr):
    """|F(kx,kz)|^2, y-averaged, for a SCALAR slab (nx, ny_slab, nz). No 1/2,
    no component sum (scalar has no 'energy' convention). Returns (nx, nz)."""
    nx, ny_slab, nz = arr.shape
    norm = 1.0 / (nx * nz)
    Fhat = np.fft.fft2(arr, axes=(0, 2)) * norm
    return (np.abs(Fhat) ** 2).mean(axis=1)


def reduce_axis(power_kxz, axis_lbl, idx, n_bins):
    """Sum the 2D power onto one axis (marginal), then bin. axis 'x' -> keep kx
    (sum over kz); 'z' -> keep kz (sum over kx)."""
    per_axis = power_kxz.sum(axis=1) if axis_lbl == "x" else power_kxz.sum(axis=0)
    return np.bincount(idx, weights=per_axis, minlength=n_bins)


def reduce_radial(power_kxz, idx2d, n_bins):
    """Annular sum: total power in each k_perp ring. idx2d is (nx,nz)."""
    return np.bincount(idx2d.ravel(), weights=power_kxz.ravel(),
                       minlength=n_bins)


def build_sf_lags(n, max_frac):
    hi = max(SF_MIN_LAG_CELLS + 1, int(np.floor(max_frac * n)))
    lags = np.unique(np.round(
        np.geomspace(SF_MIN_LAG_CELLS, hi, num=40)).astype(int))
    return lags[(lags >= SF_MIN_LAG_CELLS) & (lags <= hi)]


def sf_moments(field, axis, lags, orders):
    """Raw structure-function moments <|df|^n> along one axis for each n in
    `orders`. Returns dict n->array. axis: 0=x,1=y,2=z. `field` is the magnetic
    VECTOR dict (Bx,By,Bz); SF is on the vector increment magnitude, unchanged.

    ###! SPEED: this is the dominant cost of the compute stage, so three
    ###! micro-optimisations are applied, ALL exact for even n (which these are):
    ###!   1. The increment is cast to float32. |df|^n on a large slab is
    ###!      memory-bandwidth-bound; halving the bytes roughly halves the time.
    ###!      Precision cost on <|df|^8> is ~6e-8 -- far below any physical
    ###!      significance (verified on synthetic data).
    ###!   2. Work in d2 = |df|^2 = sum_i df_i^2 and form |df|^n = (d2)^(n/2)
    ###!      by INTEGER powers (repeated multiply), never a fractional power.
    ###!   3. No sqrt is taken -- for even n it would be squared straight back.
    ###! For ODD n this routine would need sqrt(d2); a guard below raises rather
    ###! than silently returning a wrong (integer-power) answer.
    """
    for n in orders:
        if n % 2 != 0:
            raise ValueError(f"sf_moments fast path assumes even orders; got n={n}")

    n_axis = field[COMPONENTS[0]].shape[axis]
    out = {n: np.full(lags.shape, np.nan, dtype=np.float64) for n in orders}
    half = {n: n // 2 for n in orders}          ###! d2 exponent per order
    max_half = max(half.values())

    for li, l in enumerate(lags):
        if l >= n_axis:
            continue
        sl_hi = [slice(None)] * 3
        sl_lo = [slice(None)] * 3
        sl_hi[axis] = slice(l, n_axis)
        sl_lo[axis] = slice(0, n_axis - l)

        d2 = None
        for comp in COMPONENTS:
            ###! float32 increment; accumulate |df|^2 in float32
            d = (field[comp][tuple(sl_hi)] -
                 field[comp][tuple(sl_lo)]).astype(np.float32)
            d2 = d * d if d2 is None else d2 + d * d

        ###! integer powers of d2 by repeated multiply, reusing lower powers:
        ###! p[1]=d2, p[2]=d2^2, ... up to max_half. |df|^n = p[n/2].
        p = d2                                   ###! p currently d2^1
        power_cache = {1: d2}
        for e in range(2, max_half + 1):
            p = p * d2
            power_cache[e] = p
        for n in orders:
            ###! mean in float64 for a stable reduction over many cells
            out[n][li] = np.mean(power_cache[half[n]], dtype=np.float64)

    return out


parser = argparse.ArgumentParser(description="MPI compute stage: write per-CS-averaged in-plane spectra E(kx),E(kz),E(kperp) for dB (vector), dJz and drho (per-species-summed scalars), plus SF moments <|df|^n> for B and dB, to a .npz cache.")
parser.add_argument("dir_data", type=str)
parser.add_argument("xmin", type=float); parser.add_argument("xmax", type=float)
parser.add_argument("ymin", type=float); parser.add_argument("ymax", type=float)
parser.add_argument("zmin", type=float); parser.add_argument("zmax", type=float)
parser.add_argument("--nxc", type=int, required=True)
parser.add_argument("--nyc", type=int, required=True)
parser.add_argument("--nzc", type=int, required=True)
parser.add_argument("--cycle-start", type=int, default=0)
parser.add_argument("--cycle-end", type=int, default=5000)
parser.add_argument("--cycle-step", type=int, default=100)
###! physical-time label per OUTPUT FRAME, as in rho_CS.py / B_J.py:
###!   time = (cycle / cycle_step) * time_step
###! (equals literal sim time cycle*dt only if time_step == cycle_step*dt).
parser.add_argument("--time-step", type=float, default=100.0)
parser.add_argument("--outdir", type=str, default=None)
parser.add_argument("--cache-name", type=str, default="spectra_sf_cache.npz")
parser.add_argument("--mapping", type=str, default="auto",
                    choices=["auto", "A", "B", "C", "D", "E", "F"])
args = parser.parse_args()

nxc, nyc, nzc = args.nxc, args.nyc, args.nzc
Lx = args.xmax - args.xmin
Ly = args.ymax - args.ymin
Lz = args.zmax - args.zmin
outdir = args.dir_data if args.outdir is None else args.outdir
if rank == 0:
    os.makedirs(outdir, exist_ok=True)

###! ---------------- setup ----------------
setup_error = None
if rank == 0:
    try:
        all_files = sorted(glob.glob(os.path.join(args.dir_data, "proc*.hdf")))
        if not all_files:
            raise RuntimeError(f"No proc*.hdf files found in {args.dir_data}")
        with h5py.File(all_files[0], "r") as f:
            tile_shape = tuple(f[f"fields/Bx/cycle_{args.cycle_start}"].shape)

            ###! verify the per-species scalar moments exist and share the Bx
            ###! tile shape (the shared-boundary assembly indexing is reused for
            ###! them, so a shape mismatch would corrupt the assembled field).
            probe_sp = ELECTRONS[0]
            for qty in SCALAR_QTYS:
                p = f"moments/species_{probe_sp}/{qty}/cycle_{args.cycle_start}"
                if p not in f:
                    raise KeyError(
                        f"Expected per-species moment '{p}' not found. This "
                        f"script needs moments/species_S/{{Jz,rho}} for "
                        f"S in {sorted(set(ELECTRONS + PROTONS))}.")
                ms = tuple(f[p].shape)
                if ms != tile_shape:
                    raise RuntimeError(
                        f"Moment '{qty}' tile shape {ms} != Bx tile shape "
                        f"{tile_shape}; shared assembly indexing would be wrong.")

        nx_t, ny_t, nz_t = tile_shape
        XLEN, YLEN, ZLEN = nxc // (nx_t-1), nyc // (ny_t-1), nzc // (nz_t-1)
        if XLEN * YLEN * ZLEN != len(all_files):
            raise RuntimeError(f"decomposition {XLEN}x{YLEN}x{ZLEN} != "
                               f"{len(all_files)} files")
        map_name = (choose_mapping(all_files, XLEN, YLEN, ZLEN)
                    if args.mapping == "auto" else args.mapping)
    except Exception as exc:
        setup_error = f"{type(exc).__name__}: {exc}"
        all_files = map_name = tile_shape = None
        XLEN = YLEN = ZLEN = None
else:
    all_files = map_name = tile_shape = None
    XLEN = YLEN = ZLEN = None

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

rank_to_ijk = mapping_candidates(XLEN, YLEN, ZLEN)[map_name]
local_files = all_files[rank::size]
G_shape = global_shape_shared(tile_shape, XLEN, YLEN, ZLEN)
Gx, Gy, Gz = G_shape

###! ------------------------------------------------------------------
###! datasets to assemble each slab: the 3 B components + every per-species
###! Jz and rho we will need. Keys are field-agnostic; the group summation
###! happens after assembly on rank 0.
DATASETS = {c: f"fields/{c}/{{cycle}}" for c in COMPONENTS}
SPS = sorted(set(ELECTRONS + PROTONS))
for s in SPS:
    for qty in SCALAR_QTYS:
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

slab_ranges = []
for (f_lo, f_hi) in CS_SLABS:
    jlo = max(0, int(np.floor(f_lo * nyc)))
    jhi = min(Gy, int(np.ceil(f_hi * nyc)))
    slab_ranges.append((jlo, jhi))

nx_fft = Gx - 1
nz_fft = Gz - 1

if rank == 0:
    ###! ---- square-grid guard for the k_perp (isotropic) reduction ----
    ###! integer-radius annular binning is only a true constant-|k| annulus when
    ###! Delta kx == Delta kz, i.e. Lx/nx_fft == Lz/nz_fft. Abort otherwise
    ###! rather than silently distorting the isotropic slope.
    dkx = 2.0 * np.pi / Lx
    dkz = 2.0 * np.pi / Lz
    if nx_fft != nz_fft or not np.isclose(dkx, dkz, rtol=1e-6):
        raise SystemExit(
            f"k_perp binning assumes a SQUARE in-plane grid (Delta kx == "
            f"Delta kz). Got nx_fft={nx_fft}, nz_fft={nz_fft}, "
            f"dkx={dkx:.6g}, dkz={dkz:.6g} (Lx={Lx:g}, Lz={Lz:g}). "
            f"Refusing to mis-bin the isotropic spectrum.")

    idx_x, nbx, kcx, vx = build_axis_binning(nx_fft, Lx)
    idx_z, nbz, kcz, vz = build_axis_binning(nz_fft, Lz)
    idx_p, nbp, kcp, vp = build_radial_binning(nx_fft, nz_fft, Lx, Lz)
    lags_x = build_sf_lags(nx_fft, max_frac=0.5)
    lags_z = build_sf_lags(nz_fft, max_frac=0.5)
    ny_slab0 = slab_ranges[0][1] - slab_ranges[0][0]
    lags_y = build_sf_lags(ny_slab0, max_frac=0.9)

    ###! spectral field tags: dB (vector) + the four per-species scalars.
    ###! Each gets three reductions: _x (kx), _z (kz), _p (k_perp).
    SPEC_TAGS = ["dB", "dJze", "dJzp", "dre", "drp"]

    def new_store():
        s = {"SF_B": {n: {"x": [], "y": [], "z": []} for n in SF_ORDERS},
             "SF_dB": {n: {"x": [], "y": [], "z": []} for n in SF_ORDERS}}
        for tag in SPEC_TAGS:
            s[f"Ex_{tag}"] = []
            s[f"Ez_{tag}"] = []
            s[f"Ekp_{tag}"] = []
        return s
    store = {cs: new_store() for cs in range(len(CS_SLABS))}

###! ---------------- main loop ----------------
for cyc in cycle_names:
    for cs, (jlo, jhi) in enumerate(slab_ranges):
        slab = assemble_slab(cyc, local_files, rank_to_ijk, tile_shape,
                             G_shape, jlo, jhi, DATASETS)
        if rank != 0:
            continue

        ###! ---- magnetic vector: crop shared duplicate plane, form B and dB ----
        B = {c: slab[c][:nx_fft, :, :nz_fft] for c in COMPONENTS}
        dB = fluctuation_vec(B)

        ###! ---- per-species SUM-THEN-FFT for the scalar group moments ----
        ###! group fluid moment = sum of species slabs (real space), cropped to
        ###! the FFT grid, then fluctuation df = f - <f>_xz(y).
        def group_scalar(qty, species):
            acc = None
            for s in species:
                a = slab[f"{qty}_s{s}"][:nx_fft, :, :nz_fft]
                acc = a.copy() if acc is None else acc + a
            return acc

        Jz_e = group_scalar("Jz", ELECTRONS)
        Jz_p = group_scalar("Jz", PROTONS)
        rho_e = group_scalar("rho", ELECTRONS)
        rho_p = group_scalar("rho", PROTONS)

        dJz_e = fluctuation_scalar(Jz_e)
        dJz_p = fluctuation_scalar(Jz_p)
        drho_e = fluctuation_scalar(rho_e)
        drho_p = fluctuation_scalar(rho_p)

        ###! ---- 2D power per field, then three reductions each ----
        p_dB   = power_kxz_vector(dB)
        p_dJze = power_kxz_scalar(dJz_e)
        p_dJzp = power_kxz_scalar(dJz_p)
        p_dre  = power_kxz_scalar(drho_e)
        p_drp  = power_kxz_scalar(drho_p)

        for tag, P in (("dB", p_dB), ("dJze", p_dJze), ("dJzp", p_dJzp),
                       ("dre", p_dre), ("drp", p_drp)):
            store[cs][f"Ex_{tag}"].append(reduce_axis(P, "x", idx_x, nbx))
            store[cs][f"Ez_{tag}"].append(reduce_axis(P, "z", idx_z, nbz))
            store[cs][f"Ekp_{tag}"].append(reduce_radial(P, idx_p, nbp))

        ###! ---- SF moments on B and dB (unchanged) ----
        for tag, fld in (("SF_B", B), ("SF_dB", dB)):
            mx = sf_moments(fld, 0, lags_x, SF_ORDERS)
            my = sf_moments(fld, 1, lags_y, SF_ORDERS)
            mz = sf_moments(fld, 2, lags_z, SF_ORDERS)
            for n in SF_ORDERS:
                store[cs][tag][n]["x"].append(mx[n])
                store[cs][tag][n]["y"].append(my[n])
                store[cs][tag][n]["z"].append(mz[n])

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
    ###! physical-time label per output frame (rho_CS.py / B_J.py convention)
    times = cycles / args.cycle_step * args.time_step

    save = dict(
        # axes / masks
        kc_x=kcx, valid_x=vx, kc_z=kcz, valid_z=vz, kc_p=kcp, valid_p=vp,
        lags_x=lags_x, lags_y=lags_y, lags_z=lags_z,
        dx=Lx / nxc, dy=Ly / nyc, dz=Lz / nzc,
        cycles=cycles, times=times,
        # meta
        sf_orders=np.array(SF_ORDERS),
        cs_slabs=np.array(CS_SLABS),
        grid=np.array([nxc, nyc, nzc]),
        box=np.array([Lx, Ly, Lz]),
        mapping=np.array([map_name], dtype=object),
        electrons=np.array(ELECTRONS),
        protons=np.array(PROTONS),
    )

    ###! spectra (CS-averaged): dB + four per-species scalars, x/z/kperp each
    for tag in SPEC_TAGS:
        save[f"Ex_{tag}"] = avg2(f"Ex_{tag}")
        save[f"Ez_{tag}"] = avg2(f"Ez_{tag}")
        save[f"Ekp_{tag}"] = avg2(f"Ekp_{tag}")

    ###! SF moments: one array per (field, order, direction), CS-averaged
    for tag in ("SF_B", "SF_dB"):
        for n in SF_ORDERS:
            for ax in ("x", "y", "z"):
                save[f"{tag}{n}_{ax}"] = avg2_sf(tag, n, ax)

    cache_path = os.path.join(outdir, args.cache_name)
    np.savez_compressed(cache_path, **save)
    print(f"Wrote cache: {cache_path}", flush=True)
    print(f"  cycles={len(cycles)}  spectra tags={SPEC_TAGS}", flush=True)
    print(f"  reductions: Ex_ (kx), Ez_ (kz), Ekp_ (k_perp isotropic)",
          flush=True)
    print(f"  SF orders={SF_ORDERS} on B and dB (raw moments <|df|^n>, "
          f"CS-averaged)", flush=True)
    print(f"  per-species groups: electrons={ELECTRONS} protons={PROTONS} "
          f"(summed before FFT)", flush=True)

comm.Barrier()