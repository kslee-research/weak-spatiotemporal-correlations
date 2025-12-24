import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
# Folder conventions (fixed)
# ============================================================
DIR_AUTOCORR_BASE = "./results_radial_autocorr"
DIR_CONT_BASE     = "./results_horizental"      # keep your current spelling
DIR_OUT_BASE      = "./results_compare"

# Defaults (no global mutation inside main)
DEFAULT_METRIC = "QF"        # "QF" or "tauF"
DEFAULT_AGG    = "median"    # "mean" or "median" or "p75"

# Bootstrap
DO_BOOTSTRAP = True
BOOT_N = 3000
SEED = 7


# ============================================================
# Utilities
# ============================================================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def newest_dir(pattern):
    cands = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not cands:
        return None
    return max(cands, key=lambda p: os.path.getmtime(p))

def pick_dir(base_dir, prefix, forced_name=None):
    """
    base_dir: e.g., ./results_radial_autocorr
    prefix:   e.g., run_ or patch_
    forced_name: e.g., run_20251218_221718 (folder name only)
    """
    if forced_name:
        p = os.path.join(base_dir, forced_name)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Specified folder not found: {p}")
        return p

    pattern = os.path.join(base_dir, f"{prefix}*")
    p = newest_dir(pattern)
    if p is None:
        raise FileNotFoundError(f"No folders found matching: {pattern}")
    return p

def now_out_dir(base):
    ensure_dir(base)
    rid = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out = os.path.join(base, rid)
    ensure_dir(out)
    return out

def read_csv_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        cols = {h: [] for h in header}
        for ln in f:
            if not ln.strip():
                continue
            parts = ln.strip().split(",")
            if len(parts) != len(header):
                continue
            for h, v in zip(header, parts):
                try:
                    cols[h].append(float(v))
                except:
                    cols[h].append(np.nan)
    return {k: np.array(v, dtype=float) for k, v in cols.items()}

def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 3:
        return np.nan
    x = x - x.mean(); y = y - y.mean()
    den = np.sqrt((x*x).sum() * (y*y).sum())
    if den < 1e-12:
        return np.nan
    return float((x*y).sum() / den)

def rankdata(a):
    a = np.asarray(a, float)
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    s = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j+1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j+1]] = 0.5 * (i + j)
        i = j + 1
    return ranks

def spearman(x, y):
    return pearson(rankdata(x), rankdata(y))

def bootstrap_ci(x, y, fn, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(x)
    vals = np.empty(n, dtype=float)
    for k in range(n):
        idx = rng.integers(0, N, size=N)
        vals[k] = fn(x[idx], y[idx])
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)

def interp_to(x_src, y_src, x_tgt):
    x_src = np.asarray(x_src, float)
    y_src = np.asarray(y_src, float)
    x_tgt = np.asarray(x_tgt, float)
    m = np.isfinite(x_src) & np.isfinite(y_src)
    x_src = x_src[m]; y_src = y_src[m]
    if len(x_src) < 2:
        return np.full_like(x_tgt, np.nan, dtype=float)
    o = np.argsort(x_src)
    return np.interp(x_tgt, x_src[o], y_src[o], left=np.nan, right=np.nan)

def agg_fn(x, agg):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    if agg == "mean":
        return float(np.mean(x))
    if agg == "median":
        return float(np.median(x))
    if agg == "p75":
        return float(np.percentile(x, 75))
    raise ValueError("agg must be 'mean', 'median', or 'p75'")

def safe_load_npy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path).astype(float).reshape(-1)


# ============================================================
# Main compare logic (uses r_link binning => patch_r bins)
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--autocorr", default="", help="Folder name under results_radial_autocorr (e.g., run_20251218_221718). Empty => latest.")
    ap.add_argument("--cont", default="", help="Folder name under results_horizental (e.g., patch_20251218_230454). Empty => latest.")
    ap.add_argument("--metric", default=DEFAULT_METRIC, choices=["QF", "tauF"], help="Compare against QF_link or tauF_link (default QF).")
    ap.add_argument("--agg", default=DEFAULT_AGG, choices=["mean", "median", "p75"], help="Aggregation for link->patch (default median).")
    args = ap.parse_args()

    metric = args.metric
    agg    = args.agg

    # Pick folders
    autocorr_dir = pick_dir(DIR_AUTOCORR_BASE, prefix="run_",   forced_name=args.autocorr.strip() or None)
    cont_dir     = pick_dir(DIR_CONT_BASE,     prefix="patch_", forced_name=args.cont.strip() or None)

    # Autocorr CSV path (fixed filename)
    path_autocorr_csv = os.path.join(autocorr_dir, "radial_profile.csv")
    if not os.path.exists(path_autocorr_csv):
        raise FileNotFoundError(f"Missing radial_profile.csv in: {autocorr_dir}")

    out = now_out_dir(DIR_OUT_BASE)

    # --- autocorr load ---
    A = read_csv_dict(path_autocorr_csv)
    if "r_cm" not in A or "S_peak" not in A:
        raise RuntimeError("radial_profile.csv must include columns: r_cm, S_peak")
    rA = A["r_cm"]
    S  = A["S_peak"]
    valid = A.get("valid", np.ones_like(rA))
    mA = np.isfinite(rA) & np.isfinite(S) & (valid > 0.5)
    rA = rA[mA]; S = S[mA]

    # --- continuity geometry + link metric ---
    r_patch = safe_load_npy(os.path.join(cont_dir, "patch_r_cm.npy"))
    r_link  = safe_load_npy(os.path.join(cont_dir, "r_link.npy"))

    if metric == "QF":
        link_val = safe_load_npy(os.path.join(cont_dir, "QF_link.npy"))
        yname = f"QF_link_{agg}_by_r"
    else:
        link_val = safe_load_npy(os.path.join(cont_dir, "tauF_link.npy"))
        yname = f"tauF_link_{agg}_by_r"

    if len(r_link) != len(link_val):
        raise RuntimeError(f"Length mismatch: r_link({len(r_link)}) != link_metric({len(link_val)})")

    # --- bin links into patch bins ---
    y_patch = np.full_like(r_patch, np.nan, dtype=float)

    # tolerance from patch spacing
    tol = 1e-6
    if len(r_patch) > 1:
        rs = np.sort(r_patch)
        dr = np.median(np.diff(rs))
        tol = max(1e-6, 0.51 * dr)

    for i, rp0 in enumerate(r_patch):
        m = np.abs(r_link - rp0) <= tol
        if np.any(m):
            y_patch[i] = agg_fn(link_val[m], agg)
        else:
            # fallback: nearest 10% by distance
            k = max(1, int(0.1 * len(r_link)))
            idx = np.argsort(np.abs(r_link - rp0))[:k]
            y_patch[i] = agg_fn(link_val[idx], agg)

    # interpolate y_patch onto autocorr rA grid
    y_on_A = interp_to(r_patch, y_patch, rA)
    mask = np.isfinite(S) & np.isfinite(y_on_A)
    r = rA[mask]; S2 = S[mask]; Y2 = y_on_A[mask]
    if len(r) < 5:
        raise RuntimeError("Too few aligned points after interpolation. Check r ranges / validity flags.")

    rp = pearson(S2, Y2)
    rs = spearman(S2, Y2)

    if DO_BOOTSTRAP:
        lo_p, hi_p = bootstrap_ci(S2, Y2, pearson,  n=BOOT_N, seed=SEED)
        lo_s, hi_s = bootstrap_ci(S2, Y2, spearman, n=BOOT_N, seed=SEED+1)
    else:
        lo_p = hi_p = lo_s = hi_s = np.nan

    # --- plots ---
    plt.figure()
    ax1 = plt.gca()
    ax1.plot(r, S2, marker="o", label="S_peak (autocorr persistence)")
    ax1.set_xlabel("Distance from sphere surface r (cm)")
    ax1.set_ylabel("S_peak")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(r, Y2, marker="s", linestyle="--", label=yname)
    ax2.set_ylabel(yname)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    title = f"S_peak vs {yname} (Pearson={rp:.3f}, Spearman={rs:.3f})"
    if DO_BOOTSTRAP:
        title += f"\nPearson CI[{lo_p:.2f},{hi_p:.2f}]  Spearman CI[{lo_s:.2f},{hi_s:.2f}]"
    ax1.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "overlay.png"), dpi=150)
    plt.savefig(os.path.join(out, "overlay.pdf"))
    plt.close()

    plt.figure()
    plt.scatter(S2, Y2)
    plt.xlabel("S_peak")
    plt.ylabel(yname)
    plt.grid(True, alpha=0.3)
    plt.title(f"Scatter (Pearson={rp:.3f}, Spearman={rs:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "scatter.png"), dpi=150)
    plt.savefig(os.path.join(out, "scatter.pdf"))
    plt.close()

    # --- summary ---
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"AUTOCORR_DIR: {autocorr_dir}\n")
        f.write(f"CONT_DIR: {cont_dir}\n")
        f.write(f"AUTOCORR_CSV: {path_autocorr_csv}\n")
        f.write(f"METRIC: {metric}\n")
        f.write(f"AGG: {agg}\n")
        f.write(f"N aligned: {len(r)}\n")
        f.write(f"Pearson: {rp:.6f}\n")
        if DO_BOOTSTRAP:
            f.write(f"Pearson 95% CI: [{lo_p:.6f}, {hi_p:.6f}]\n")
        f.write(f"Spearman: {rs:.6f}\n")
        if DO_BOOTSTRAP:
            f.write(f"Spearman 95% CI: [{lo_s:.6f}, {hi_s:.6f}]\n")

    print("=============================================")
    print("COMPARE DONE")
    print("AUTOCORR:", autocorr_dir)
    print("CONT    :", cont_dir)
    print("OUT     :", out)
    print("Metric  :", metric, "AGG:", agg)
    print("Pearson :", rp, "Spearman:", rs)
    if DO_BOOTSTRAP:
        print("Pearson CI :", (lo_p, hi_p))
        print("Spearman CI:", (lo_s, hi_s))
    print("=============================================")


if __name__ == "__main__":
    main()
