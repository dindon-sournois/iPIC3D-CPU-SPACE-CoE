"""
    srun python3 -u "$SCRIPT" "$DATA_DIR" \
        $xmin $xmax $ymin $ymax $zmin $zmax \
        --nxc "$nxc" --nyc "$nyc" --nzc "$nzc" \
        --cycle-start 0 --cycle-end 20000 --cycle-step 100 --time-step 20 \
        --outdir "$OUT_DIR"

    Cache written: <outdir>/spectra_sf_cache.npz  (override with --cache-name).
"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import gc
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

###! float32 for the assembled real-space plane. Halves memory + bandwidth on
###! the largest arrays. All spectral quantities are fluctuations (mean removed)
###! or |k|-space powers, where float32 error (~1e-7 relative) is far below PIC
###! counting noise. The SF path already used float32 internally.
ASM_DTYPE = np.float32                  ###! assembly / accumulation dtype

###! ------------------------------------------------------------------
###! Species groupings (0-indexed), matching rho_CS.py / B_J.py conventions.
ELECTRONS = [0, 2]
PROTONS   = [1, 3]

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
###! Shared-geometry count buffer. Every field in a slab lands on exactly the
###! same nodes with the same duplicate-plane overlap, so the per-node write
###! count depends ONLY on (cycle-independent) tile geometry and the slab
###! [jlo,jhi). We compute it ONCE per slab and reuse it for every field,
###! instead of recomputing an identical cnt array inside each assembly.
###! ------------------------------------------------------------------

def assemble_count(local_files, rank_to_ijk, tile_shape, G_shape, jlo, jhi):
    """COLLECTIVE. Return, on rank 0, the (Gx,ny_slab,Gz) float64 count of how
    many tiles wrote each node in y-slab [jlo,jhi); None elsewhere. Reused as
    the divisor for every field of this slab."""
    Gx, Gy, Gz = G_shape
    nx_t, ny_t, nz_t = tile_shape
    nx_c, ny_c, nz_c = nx_t - 1, ny_t - 1, nz_t - 1
    ny_slab = jhi - jlo

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
        oy = a - jlo
        ny_read = b - a + 1
        cnt[gx0:gx0+nxu, oy:oy+ny_read, gz0:gz0+nzu] += 1.0
    comm.Allreduce(MPI.IN_PLACE, cnt, op=MPI.SUM)
    return cnt if rank == 0 else None


def assemble_one(cycle_name, path_tmpl, local_files, rank_to_ijk, tile_shape,
                 G_shape, jlo, jhi, cnt):
    """COLLECTIVE. Assemble a SINGLE field (given by path_tmpl.format(
    cycle=cycle_name)) on (x,z) over y-slab [jlo,jhi); reduce to rank 0 and
    average using the precomputed `cnt` divisor. Returns (Gx,ny_slab,Gz)
    ASM_DTYPE array on rank 0, else None.

    ###! Memory: only ONE such buffer exists at a time (plus the shared cnt),
    ###! versus the original which held one per field for all fields at once."""
    Gx, Gy, Gz = G_shape
    nx_t, ny_t, nz_t = tile_shape
    nx_c, ny_c, nz_c = nx_t - 1, ny_t - 1, nz_t - 1
    ny_slab = jhi - jlo

    acc = np.zeros((Gx, ny_slab, Gz), dtype=ASM_DTYPE)
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
        path = path_tmpl.format(cycle=cycle_name)
        with h5py.File(fp, "r") as f:
            if path not in f:
                raise KeyError(
                    f"Missing dataset {path} in {os.path.basename(fp)}")
            ###! read native dtype, cast to ASM_DTYPE (float32) on assignment;
            ###! avoids a float64 temporary of the whole block.
            blk = f[path][xs:, js:je, zs:]
            acc[gx0:gx0+nxu, oy:oy+ny_read, gz0:gz0+nzu] += blk

    ###! reduce in the assembly dtype to keep the message half-size. float32
    ###! summation of a partition (each node summed on exactly one owning rank
    ###! set) incurs no extra rounding beyond the per-tile add already done.
    comm.Allreduce(MPI.IN_PLACE, acc, op=MPI.SUM)
    if rank != 0:
        return None
    ok = cnt > 0
    ###! divide float32 field by float64 count -> stays float32
    acc[ok] /= cnt[ok].astype(ASM_DTYPE)
    if not ok.all():
        print(f"  WARNING: {int((~ok).sum())} slab nodes never written at "
              f"{cycle_name}", flush=True)
    return acc


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
    REQUIRES a square grid (Delta kx == Delta kz); guarded in __main__."""
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=Lz / nz)
    KX, KZ = np.meshgrid(kx, kz, indexing="ij")
    kperp = np.sqrt(KX * KX + KZ * KZ)

    k0 = 2.0 * np.pi / Lx
    idx2d = np.floor(kperp / k0 + 0.5).astype(np.int64)
    n_bins = int(idx2d.max()) + 1

    flat_idx = idx2d.ravel()
    flat_k = kperp.ravel()
    counts = np.bincount(flat_idx, minlength=n_bins)
    k_sum = np.bincount(flat_idx, weights=flat_k, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        kc = np.where(counts > 0, k_sum / np.maximum(counts, 1), np.nan)

    kny = min(np.pi * nx / Lx, np.pi * nz / Lz)
    valid = (counts > 0) & (kc <= kny) & (np.arange(n_bins) > 0)
    return idx2d, n_bins, kc, valid


def power_kxz_vector(field):
    """0.5 * sum_i |F_i(kx,kz)|^2, y-averaged. `field` is a dict of 3 real
    slabs. Returns (nx, nz) 2D power.

    ###! rfft would halve the FFT work/memory, but the downstream marginal and
    ###! annular reductions index the FULL (nx,nz) fftfreq grid built at setup;
    ###! switching to rfft would require rebuilding those indexers for the
    ###! half-spectrum and doubling non-Nyquist modes. Kept as full fft2 to
    ###! preserve identical binning. The FFT input is float32 so numpy returns
    ###! complex64 -- already half the memory of the original complex128."""
    nx, ny_slab, nz = field[COMPONENTS[0]].shape
    norm = 1.0 / (nx * nz)
    power = np.zeros((nx, nz), dtype=np.float64)
    for comp in COMPONENTS:
        Fhat = np.fft.fft2(field[comp], axes=(0, 2)) * norm
        power += 0.5 * (np.abs(Fhat) ** 2).mean(axis=1)
        del Fhat                                   ###! free per-component FFT
    return power


def power_kxz_scalar(arr):
    """|F(kx,kz)|^2, y-averaged, for a SCALAR slab. Returns (nx, nz)."""
    nx, ny_slab, nz = arr.shape
    norm = 1.0 / (nx * nz)
    Fhat = np.fft.fft2(arr, axes=(0, 2)) * norm
    out = (np.abs(Fhat) ** 2).mean(axis=1)
    del Fhat
    return out


def reduce_axis(power_kxz, axis_lbl, idx, n_bins):
    per_axis = power_kxz.sum(axis=1) if axis_lbl == "x" else power_kxz.sum(axis=0)
    return np.bincount(idx, weights=per_axis, minlength=n_bins)


def reduce_radial(power_kxz, idx2d, n_bins):
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
    VECTOR dict (Bx,By,Bz); SF is on the vector increment magnitude.

    ###! Even-order fast path (unchanged math): work in d2 = |df|^2 and form
    ###! |df|^n = (d2)^(n/2) by integer powers, no sqrt.
    ###! MEMORY CHANGE: the original cached EVERY intermediate power
    ###! d2^1..d2^max_half simultaneously (power_cache), i.e. up to max_half
    ###! full slab copies. Here we keep only a single running product `p` and
    ###! reduce each requested order as p climbs through the needed exponents,
    ###! so peak is d2 + p + one increment temporary ~= 3 slabs regardless of
    ###! how many/high the orders are. Results are bit-identical.
    """
    for n in orders:
        if n % 2 != 0:
            raise ValueError(f"sf_moments fast path assumes even orders; got n={n}")

    n_axis = field[COMPONENTS[0]].shape[axis]
    out = {n: np.full(lags.shape, np.nan, dtype=np.float64) for n in orders}
    half = {n: n // 2 for n in orders}          ###! d2 exponent per order
    ###! map exponent -> list of orders that want it, so we emit at the right
    ###! step of the running product without storing all powers.
    want_at = {}
    for n in orders:
        want_at.setdefault(half[n], []).append(n)
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
            d = (field[comp][tuple(sl_hi)] -
                 field[comp][tuple(sl_lo)]).astype(np.float32)
            d2 = d * d if d2 is None else d2 + d * d
            del d

        ###! running product p = d2^e; emit any orders whose half == e.
        p = d2                                   ###! e = 1
        if 1 in want_at:
            for n in want_at[1]:
                out[n][li] = np.mean(p, dtype=np.float64)
        for e in range(2, max_half + 1):
            p = p * d2
            if e in want_at:
                for n in want_at[e]:
                    out[n][li] = np.mean(p, dtype=np.float64)
        del d2, p

    return out


parser = argparse.ArgumentParser(description="MPI compute stage: write per-CS-averaged in-plane spectra E(kx),E(kz),E(kperp) for dB (vector), dJz and drho (per-species-summed scalars), plus SF moments <|df|^n> for B and dB, to a .npz cache. Memory-optimised field-serial assembly.")
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

###! per-field HDF5 path templates: 3 B components always, plus only the
###! per-species Jz/rho that were found available.
B_PATHS = {c: f"fields/{c}/{{cycle}}" for c in COMPONENTS}
needed_species_qty = set()
for tag, (qty, members) in tag_source.items():
    for s in members:
        needed_species_qty.add((s, qty))
SPECIES_PATHS = {(s, qty): f"moments/species_{s}/{qty}/{{cycle}}"
                 for (s, qty) in sorted(needed_species_qty)}

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

    def new_store():
        s = {"SF_B": {n: {"x": [], "y": [], "z": []} for n in SF_ORDERS},
             "SF_dB": {n: {"x": [], "y": [], "z": []} for n in SF_ORDERS}}
        for tag in spec_tags:
            s[f"Ex_{tag}"] = []
            s[f"Ez_{tag}"] = []
            s[f"Ekp_{tag}"] = []
        return s
    store = {cs: new_store() for cs in range(len(CS_SLABS))}

###! ---------------- main loop ----------------
###! Per (cycle, CS slab) the sequence is now field-serial:
###!   1. compute the shared count buffer ONCE.
###!   2. assemble Bx, By, Bz one at a time into a small dict; the three B
###!      slabs are needed together for the vector power / SF, so they do
###!      co-reside -- 3 slabs, not 8. dB is derived, reductions taken, then
###!      B is freed before touching moments.
###!   3. for each available scalar group: assemble & sum its species one at a
###!      time (never more than 2 species slabs live), reduce, free.
###! Peak real-space residency: max(3 B slabs, 2 species slabs) + cnt.
for cyc in cycle_names:
    for cs, (jlo, jhi) in enumerate(slab_ranges):

        cnt = assemble_count(local_files, rank_to_ijk, tile_shape,
                             G_shape, jlo, jhi)

        ###! ---- magnetic vector (3 co-resident slabs) ----
        B = {}
        for c in COMPONENTS:
            full = assemble_one(cyc, B_PATHS[c], local_files, rank_to_ijk,
                                tile_shape, G_shape, jlo, jhi, cnt)
            if rank == 0:
                ###! crop shared duplicate plane to the FFT grid immediately,
                ###! then drop the full-plane buffer.
                B[c] = full[:nx_fft, :, :nz_fft].copy()
                del full

        if rank == 0:
            dB = {c: B[c] - B[c].mean(axis=(0, 2), keepdims=True)
                  for c in COMPONENTS}

            p_dB = power_kxz_vector(dB)
            store[cs]["Ex_dB"].append(reduce_axis(p_dB, "x", idx_x, nbx))
            store[cs]["Ez_dB"].append(reduce_axis(p_dB, "z", idx_z, nbz))
            store[cs]["Ekp_dB"].append(reduce_radial(p_dB, idx_p, nbp))
            del p_dB

            ###! SF on B and dB (needs the three components together)
            for tag, fld in (("SF_B", B), ("SF_dB", dB)):
                mx = sf_moments(fld, 0, lags_x, SF_ORDERS)
                my = sf_moments(fld, 1, lags_y, SF_ORDERS)
                mz = sf_moments(fld, 2, lags_z, SF_ORDERS)
                for n in SF_ORDERS:
                    store[cs][tag][n]["x"].append(mx[n])
                    store[cs][tag][n]["y"].append(my[n])
                    store[cs][tag][n]["z"].append(mz[n])
            del dB, B
            gc.collect()

        ###! ---- per-species scalar groups (<=2 co-resident slabs) ----
        for tag, (qty, members) in tag_source.items():
            acc = None
            for s in members:
                full = assemble_one(cyc, SPECIES_PATHS[(s, qty)], local_files,
                                    rank_to_ijk, tile_shape, G_shape,
                                    jlo, jhi, cnt)
                if rank == 0:
                    piece = full[:nx_fft, :, :nz_fft]
                    acc = piece.copy() if acc is None else acc + piece
                    del full, piece
            if rank == 0:
                df = acc - acc.mean(axis=(0, 2), keepdims=True)
                del acc
                P = power_kxz_scalar(df)
                del df
                store[cs][f"Ex_{tag}"].append(reduce_axis(P, "x", idx_x, nbx))
                store[cs][f"Ez_{tag}"].append(reduce_axis(P, "z", idx_z, nbz))
                store[cs][f"Ekp_{tag}"].append(reduce_radial(P, idx_p, nbp))
                del P

        if rank == 0:
            del cnt
            gc.collect()

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
        kc_x=kcx, valid_x=vx, kc_z=kcz, valid_z=vz, kc_p=kcp, valid_p=vp,
        lags_x=lags_x, lags_y=lags_y, lags_z=lags_z,
        dx=Lx / nxc, dy=Ly / nyc, dz=Lz / nzc,
        cycles=cycles, times=times,
        sf_orders=np.array(SF_ORDERS),
        cs_slabs=np.array(CS_SLABS),
        grid=np.array([nxc, nyc, nzc]),
        box=np.array([Lx, Ly, Lz]),
        mapping=np.array([map_name], dtype=object),
        electrons=np.array(ELECTRONS),
        protons=np.array(PROTONS),
        spec_tags=np.array(spec_tags, dtype=object),
    )

    for tag in spec_tags:
        save[f"Ex_{tag}"] = avg2(f"Ex_{tag}")
        save[f"Ez_{tag}"] = avg2(f"Ez_{tag}")
        save[f"Ekp_{tag}"] = avg2(f"Ekp_{tag}")

    for tag in ("SF_B", "SF_dB"):
        for n in SF_ORDERS:
            for ax in ("x", "y", "z"):
                save[f"{tag}{n}_{ax}"] = avg2_sf(tag, n, ax)

    cache_path = os.path.join(outdir, args.cache_name)
    np.savez_compressed(cache_path, **save)
    print(f"Wrote cache: {cache_path}", flush=True)
    print(f"  cycles={len(cycles)}  spectra tags={spec_tags}", flush=True)
    print(f"  reductions: Ex_ (kx), Ez_ (kz), Ekp_ (k_perp isotropic)",
          flush=True)
    print(f"  SF orders={SF_ORDERS} on B and dB (raw moments <|df|^n>, "
          f"CS-averaged)", flush=True)
    print(f"  per-species groups: electrons={ELECTRONS} protons={PROTONS} "
          f"(summed before FFT)", flush=True)
    print(f"  assembly dtype={np.dtype(ASM_DTYPE).name} (field-serial)",
          flush=True)

comm.Barrier()