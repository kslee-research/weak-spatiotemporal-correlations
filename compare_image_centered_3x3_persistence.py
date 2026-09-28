# compare_image_centered_3x3_persistence_v2.py
# Paper 4: extended comparison for image-centered 3x3 temporal persistence
#
# Requires analyzer v2 outputs:
#   spatial_profile.csv
#   autocorr_curves.csv
#   temporal_metrics.csv
#
# Adds:
#   - run-level mean S_peak
#   - within-run SD across 9 patches
#   - long-lag positive AUC (0.5-2.0 s)
#   - long-lag signed AUC
#   - decay time to ACF <= 0.10
#   - material-level ensemble mean autocorrelation curves
#
# Output:
#   results_center_referenced_compare/compare_v2_YYYYmmdd_HHMMSS/

import os
import csv
import math
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 0) Matplotlib
# ============================================================
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 150

# ============================================================
# 1) Paths/config
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_ROOT = os.path.join(BASE_DIR, "results_center_referenced_radial_autocorr")
OUTPUT_ROOT = os.path.join(BASE_DIR, "results_center_referenced_compare")

SPATIAL_FILE = "spatial_profile.csv"
ACF_FILE = "autocorr_curves.csv"

TARGET_SPHERES = ("steel", "tungsten")
PATCHES = tuple(f"P{i}" for i in range(1, 10))

LONG_LAG_START_SEC = 0.50
LONG_LAG_END_SEC = 2.00
DECAY_THRESHOLD = 0.10

# ============================================================
# 2) Helpers
# ============================================================
def canonical_sphere_type(raw):
    s = (raw or "").strip().lower()
    aliases = {
        "s": "steel", "steel": "steel", "stainless": "steel",
        "stainless steel": "steel", "ss": "steel",
        "t": "tungsten", "w": "tungsten",
        "tungsten": "tungsten", "wolfram": "tungsten",
    }
    return aliases.get(s, s)


def to_float(v):
    try:
        if v is None or str(v).strip() == "":
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def finite(x):
    return np.isfinite(x)


def fmt(x, digits=8):
    return "" if not finite(x) else f"{float(x):.{digits}f}"


def mean_sd_sem(values):
    a = np.asarray([x for x in values if finite(x)], dtype=float)
    n = int(a.size)
    if n == 0:
        return np.nan, np.nan, np.nan, 0
    m = float(np.mean(a))
    if n >= 2:
        sd = float(np.std(a, ddof=1))
        sem = sd / math.sqrt(n)
    else:
        sd = np.nan
        sem = np.nan
    return m, sd, sem, n


def write_csv(path, fieldnames, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def find_run_dirs(root):
    out = []
    if not os.path.isdir(root):
        return out
    for dirpath, _, files in os.walk(root):
        if SPATIAL_FILE in files:
            out.append(dirpath)
    return sorted(out)


def read_spatial(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            sphere = canonical_sphere_type(r.get("sphere_type"))
            if sphere not in TARGET_SPHERES:
                continue
            pid = (r.get("patch_id") or "").strip().upper()
            if pid not in PATCHES:
                continue
            rows.append({
                "run_id": (r.get("run_id") or "").strip(),
                "sphere_type": sphere,
                "patch_id": pid,
                "S_peak": to_float(r.get("S_peak")),
                "long_lag_positive_auc": to_float(r.get("long_lag_positive_auc")),
                "long_lag_signed_auc": to_float(r.get("long_lag_signed_auc")),
                "decay_time_to_0p1_sec": to_float(r.get("decay_time_to_0p1_sec")),
            })
    return rows


def read_acf(path):
    tau = []
    vals = {p: [] for p in PATCHES}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f)
        for r in rd:
            tau.append(to_float(r.get("lag_sec")))
            for p in PATCHES:
                vals[p].append(to_float(r.get(p)))
    tau = np.asarray(tau, dtype=float)
    vals = {p: np.asarray(v, dtype=float) for p, v in vals.items()}
    return tau, vals


def derive_metrics_from_acf(tau, ac):
    mask = np.isfinite(tau) & np.isfinite(ac)
    tau = tau[mask]
    ac = ac[mask]
    if tau.size == 0:
        return np.nan, np.nan, np.nan

    m = (tau >= LONG_LAG_START_SEC) & (tau <= LONG_LAG_END_SEC)
    if np.sum(m) >= 2:
        pos_auc = float(np.trapz(np.maximum(ac[m], 0.0), tau[m]))
        signed_auc = float(np.trapz(ac[m], tau[m]))
    else:
        pos_auc = np.nan
        signed_auc = np.nan

    ix = np.where((tau >= 0.10) & (ac <= DECAY_THRESHOLD))[0]
    decay = float(tau[ix[0]]) if ix.size else np.nan
    return pos_auc, signed_auc, decay


def interp_curve(tau_src, y_src, tau_target):
    mask = np.isfinite(tau_src) & np.isfinite(y_src)
    if np.sum(mask) < 2:
        return np.full_like(tau_target, np.nan, dtype=float)
    return np.interp(tau_target, tau_src[mask], y_src[mask])


# ============================================================
# 3) Main
# ============================================================
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    out_dir = os.path.join(
        OUTPUT_ROOT,
        datetime.now().strftime("compare_v2_%Y%m%d_%H%M%S")
    )
    os.makedirs(out_dir, exist_ok=False)

    log_path = os.path.join(out_dir, "comparison_log.txt")
    with open(log_path, "w", encoding="utf-8") as log:
        def logw(s):
            line = f"[{datetime.now().strftime('%H:%M:%S')}] {s}"
            print(line)
            log.write(line + "\n")
            log.flush()

        run_dirs = find_run_dirs(INPUT_ROOT)
        logw(f"Found {len(run_dirs)} run folder(s).")

        runs = []
        skipped = []

        for d in run_dirs:
            sp = os.path.join(d, SPATIAL_FILE)
            acp = os.path.join(d, ACF_FILE)

            try:
                rows = read_spatial(sp)
                if not rows:
                    raise RuntimeError("No valid spatial rows.")

                run_id = rows[0]["run_id"]
                sphere = rows[0]["sphere_type"]

                if not os.path.exists(acp):
                    skipped.append((d, "missing autocorr_curves.csv"))
                    logw(f"SKIP {os.path.basename(d)}: missing autocorr_curves.csv")
                    continue

                tau, acf = read_acf(acp)

                # ensure temporal metrics exist even if old spatial_profile lacks them
                by_patch = {r["patch_id"]: r for r in rows}
                for p in PATCHES:
                    if p not in by_patch:
                        continue
                    pos, signed, decay = derive_metrics_from_acf(tau, acf[p])

                    if not finite(by_patch[p]["long_lag_positive_auc"]):
                        by_patch[p]["long_lag_positive_auc"] = pos
                    if not finite(by_patch[p]["long_lag_signed_auc"]):
                        by_patch[p]["long_lag_signed_auc"] = signed
                    if not finite(by_patch[p]["decay_time_to_0p1_sec"]):
                        by_patch[p]["decay_time_to_0p1_sec"] = decay

                runs.append({
                    "run_id": run_id,
                    "sphere_type": sphere,
                    "rows": list(by_patch.values()),
                    "tau": tau,
                    "acf": acf,
                    "source_dir": d,
                })
                logw(f"Loaded {sphere}: {run_id}")

            except Exception as e:
                skipped.append((d, str(e)))
                logw(f"SKIP {os.path.basename(d)}: {e}")

        if not runs:
            raise RuntimeError(
                "No v2-compatible runs found. Re-run videos with analyzer v2."
            )

        by_sphere = {s: [] for s in TARGET_SPHERES}
        for r in runs:
            by_sphere[r["sphere_type"]].append(r)

        for s in TARGET_SPHERES:
            logw(f"{s}: {len(by_sphere[s])} run(s)")

        # --------------------------------------------------------
        # Run-level summary
        # --------------------------------------------------------
        run_rows = []
        metrics_by_sphere = {
            s: {
                "mean_S": [], "within_SD": [],
                "AUC_pos": [], "AUC_signed": [], "decay": []
            } for s in TARGET_SPHERES
        }

        for run in runs:
            vals = defaultdict(list)
            for row in run["rows"]:
                vals["S"].append(row["S_peak"])
                vals["AUC_pos"].append(row["long_lag_positive_auc"])
                vals["AUC_signed"].append(row["long_lag_signed_auc"])
                vals["decay"].append(row["decay_time_to_0p1_sec"])

            S = np.asarray([x for x in vals["S"] if finite(x)], dtype=float)
            Ap = np.asarray([x for x in vals["AUC_pos"] if finite(x)], dtype=float)
            As = np.asarray([x for x in vals["AUC_signed"] if finite(x)], dtype=float)
            Dt = np.asarray([x for x in vals["decay"] if finite(x)], dtype=float)

            mean_S = float(np.mean(S)) if S.size else np.nan
            within_sd = float(np.std(S, ddof=1)) if S.size >= 2 else np.nan
            mean_Ap = float(np.mean(Ap)) if Ap.size else np.nan
            mean_As = float(np.mean(As)) if As.size else np.nan
            mean_Dt = float(np.mean(Dt)) if Dt.size else np.nan

            s = run["sphere_type"]
            metrics_by_sphere[s]["mean_S"].append(mean_S)
            metrics_by_sphere[s]["within_SD"].append(within_sd)
            metrics_by_sphere[s]["AUC_pos"].append(mean_Ap)
            metrics_by_sphere[s]["AUC_signed"].append(mean_As)
            metrics_by_sphere[s]["decay"].append(mean_Dt)

            run_rows.append({
                "run_id": run["run_id"],
                "sphere_type": s,
                "n_patches": int(S.size),
                "mean_S_peak": fmt(mean_S),
                "within_run_SD_S_peak": fmt(within_sd),
                "mean_long_lag_positive_AUC": fmt(mean_Ap),
                "mean_long_lag_signed_AUC": fmt(mean_As),
                "mean_decay_time_to_0p1_sec": fmt(mean_Dt),
            })

        write_csv(
            os.path.join(out_dir, "run_level_extended_metrics.csv"),
            list(run_rows[0].keys()),
            run_rows
        )

        # --------------------------------------------------------
        # Material-level summary
        # --------------------------------------------------------
        mat_rows = []
        material_stats = {}

        for s in TARGET_SPHERES:
            material_stats[s] = {}
            row = {"sphere_type": s, "n_runs": len(by_sphere[s])}

            for key, label in [
                ("mean_S", "run_mean_S_peak"),
                ("within_SD", "within_run_SD_S_peak"),
                ("AUC_pos", "long_lag_positive_AUC"),
                ("AUC_signed", "long_lag_signed_AUC"),
                ("decay", "decay_time_to_0p1_sec"),
            ]:
                m, sd, sem, n = mean_sd_sem(metrics_by_sphere[s][key])
                material_stats[s][key] = {
                    "mean": m, "sd": sd, "sem": sem, "n": n
                }
                row[f"mean_{label}"] = fmt(m)
                row[f"sd_{label}"] = fmt(sd)
                row[f"sem_{label}"] = fmt(sem)

            # run-to-run CV for mean S
            m = material_stats[s]["mean_S"]["mean"]
            sd = material_stats[s]["mean_S"]["sd"]
            cv = sd / m if finite(m) and finite(sd) and abs(m) > 1e-15 else np.nan
            row["cv_run_mean_S_peak"] = fmt(cv)
            row["cv_run_mean_S_peak_percent"] = fmt(cv * 100.0, 4) if finite(cv) else ""

            mat_rows.append(row)

        write_csv(
            os.path.join(out_dir, "material_extended_summary.csv"),
            list(mat_rows[0].keys()),
            mat_rows
        )

        # --------------------------------------------------------
        # Patch-level summary for all metrics
        # --------------------------------------------------------
        patch_rows = []
        patch_stats = {}

        for s in TARGET_SPHERES:
            for p in PATCHES:
                selected = []
                for run in by_sphere[s]:
                    for r in run["rows"]:
                        if r["patch_id"] == p:
                            selected.append(r)
                            break

                stats = {}
                for key in [
                    "S_peak", "long_lag_positive_auc",
                    "long_lag_signed_auc", "decay_time_to_0p1_sec"
                ]:
                    m, sd, sem, n = mean_sd_sem([r[key] for r in selected])
                    stats[key] = {"mean": m, "sd": sd, "sem": sem, "n": n}

                patch_stats[(s, p)] = stats

                patch_rows.append({
                    "sphere_type": s,
                    "patch_id": p,
                    "n_runs": stats["S_peak"]["n"],
                    "mean_S_peak": fmt(stats["S_peak"]["mean"]),
                    "sd_S_peak": fmt(stats["S_peak"]["sd"]),
                    "mean_long_lag_positive_AUC": fmt(stats["long_lag_positive_auc"]["mean"]),
                    "sd_long_lag_positive_AUC": fmt(stats["long_lag_positive_auc"]["sd"]),
                    "mean_long_lag_signed_AUC": fmt(stats["long_lag_signed_auc"]["mean"]),
                    "sd_long_lag_signed_AUC": fmt(stats["long_lag_signed_auc"]["sd"]),
                    "mean_decay_time_to_0p1_sec": fmt(stats["decay_time_to_0p1_sec"]["mean"]),
                    "sd_decay_time_to_0p1_sec": fmt(stats["decay_time_to_0p1_sec"]["sd"]),
                })

        write_csv(
            os.path.join(out_dir, "patch_extended_summary.csv"),
            list(patch_rows[0].keys()),
            patch_rows
        )

        # --------------------------------------------------------
        # Ensemble mean autocorrelation curves
        # --------------------------------------------------------
        all_tau = [run["tau"] for run in runs if run["tau"].size]
        max_common = min(float(np.nanmax(t)) for t in all_tau)
        # use the densest available time step among runs, but not zero
        dts = []
        for t in all_tau:
            u = np.diff(t[np.isfinite(t)])
            u = u[u > 0]
            if u.size:
                dts.append(float(np.median(u)))
        dt = min(dts) if dts else 1.0 / 29.97
        tau_common = np.arange(0.0, max_common + 0.5 * dt, dt)

        ensemble_curves = {}
        ensemble_sd = {}

        for s in TARGET_SPHERES:
            run_mean_curves = []
            for run in by_sphere[s]:
                patch_curves = []
                for p in PATCHES:
                    patch_curves.append(
                        interp_curve(run["tau"], run["acf"][p], tau_common)
                    )
                patch_curves = np.vstack(patch_curves)
                run_mean_curves.append(np.nanmean(patch_curves, axis=0))

            if run_mean_curves:
                mat = np.vstack(run_mean_curves)
                ensemble_curves[s] = np.nanmean(mat, axis=0)
                ensemble_sd[s] = (
                    np.nanstd(mat, axis=0, ddof=1)
                    if mat.shape[0] >= 2
                    else np.zeros(mat.shape[1], dtype=float)
                )
            else:
                ensemble_curves[s] = np.full_like(tau_common, np.nan)
                ensemble_sd[s] = np.full_like(tau_common, np.nan)

        acf_rows = []
        for i, t in enumerate(tau_common):
            acf_rows.append({
                "lag_sec": fmt(t, 8),
                "steel_mean_acf": fmt(ensemble_curves["steel"][i]),
                "steel_sd_acf": fmt(ensemble_sd["steel"][i]),
                "tungsten_mean_acf": fmt(ensemble_curves["tungsten"][i]),
                "tungsten_sd_acf": fmt(ensemble_sd["tungsten"][i]),
            })

        write_csv(
            os.path.join(out_dir, "ensemble_mean_autocorrelation.csv"),
            list(acf_rows[0].keys()),
            acf_rows
        )

        # --------------------------------------------------------
        # Figures
        # --------------------------------------------------------

        # 1. Run mean S_peak
        fig = plt.figure(figsize=(7.2, 5.2))
        for xi, s in enumerate(TARGET_SPHERES):
            vals = np.asarray(metrics_by_sphere[s]["mean_S"], dtype=float)
            vals = vals[np.isfinite(vals)]
            jit = np.linspace(-0.08, 0.08, max(len(vals), 1))[:len(vals)]
            plt.scatter(np.full(len(vals), xi) + jit, vals, s=45, label=f"{s.capitalize()} runs")
            st = material_stats[s]["mean_S"]
            if finite(st["mean"]):
                plt.errorbar(
                    [xi], [st["mean"]],
                    yerr=[0 if not finite(st["sd"]) else st["sd"]],
                    marker="D", capsize=6, linewidth=1.8, markersize=7
                )
        plt.xticks([0, 1], ["Steel", "Tungsten"])
        plt.ylabel("Run mean S_peak")
        plt.title("Run-level ensemble persistence")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "run_mean_S_peak.png"), dpi=180)
        plt.savefig(os.path.join(out_dir, "run_mean_S_peak.pdf"))
        plt.close(fig)

        # 2. Within-run SD
        fig = plt.figure(figsize=(7.2, 5.2))
        for xi, s in enumerate(TARGET_SPHERES):
            vals = np.asarray(metrics_by_sphere[s]["within_SD"], dtype=float)
            vals = vals[np.isfinite(vals)]
            jit = np.linspace(-0.08, 0.08, max(len(vals), 1))[:len(vals)]
            plt.scatter(np.full(len(vals), xi) + jit, vals, s=45)
            st = material_stats[s]["within_SD"]
            if finite(st["mean"]):
                plt.errorbar(
                    [xi], [st["mean"]],
                    yerr=[0 if not finite(st["sd"]) else st["sd"]],
                    marker="D", capsize=6, linewidth=1.8, markersize=7
                )
        plt.xticks([0, 1], ["Steel", "Tungsten"])
        plt.ylabel("Within-run SD of 9 patch S_peak")
        plt.title("Spatial variability within each run")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "within_run_spatial_variability.png"), dpi=180)
        plt.savefig(os.path.join(out_dir, "within_run_spatial_variability.pdf"))
        plt.close(fig)

        # 3. Long-lag positive AUC
        fig = plt.figure(figsize=(7.2, 5.2))
        for xi, s in enumerate(TARGET_SPHERES):
            vals = np.asarray(metrics_by_sphere[s]["AUC_pos"], dtype=float)
            vals = vals[np.isfinite(vals)]
            jit = np.linspace(-0.08, 0.08, max(len(vals), 1))[:len(vals)]
            plt.scatter(np.full(len(vals), xi) + jit, vals, s=45)
            st = material_stats[s]["AUC_pos"]
            if finite(st["mean"]):
                plt.errorbar(
                    [xi], [st["mean"]],
                    yerr=[0 if not finite(st["sd"]) else st["sd"]],
                    marker="D", capsize=6, linewidth=1.8, markersize=7
                )
        plt.xticks([0, 1], ["Steel", "Tungsten"])
        plt.ylabel("Mean positive ACF area (0.5-2.0 s)")
        plt.title("Long-lag positive persistence")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "long_lag_positive_AUC.png"), dpi=180)
        plt.savefig(os.path.join(out_dir, "long_lag_positive_AUC.pdf"))
        plt.close(fig)

        # 4. Decay time
        fig = plt.figure(figsize=(7.2, 5.2))
        for xi, s in enumerate(TARGET_SPHERES):
            vals = np.asarray(metrics_by_sphere[s]["decay"], dtype=float)
            vals = vals[np.isfinite(vals)]
            jit = np.linspace(-0.08, 0.08, max(len(vals), 1))[:len(vals)]
            plt.scatter(np.full(len(vals), xi) + jit, vals, s=45)
            st = material_stats[s]["decay"]
            if finite(st["mean"]):
                plt.errorbar(
                    [xi], [st["mean"]],
                    yerr=[0 if not finite(st["sd"]) else st["sd"]],
                    marker="D", capsize=6, linewidth=1.8, markersize=7
                )
        plt.xticks([0, 1], ["Steel", "Tungsten"])
        plt.ylabel(f"Mean decay time to ACF ≤ {DECAY_THRESHOLD:.2f} (s)")
        plt.title("Autocorrelation decay time")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "decay_time_comparison.png"), dpi=180)
        plt.savefig(os.path.join(out_dir, "decay_time_comparison.pdf"))
        plt.close(fig)

        # 5. Ensemble mean ACF with ±1 SD ribbon
        fig = plt.figure(figsize=(8.2, 5.6))
        for s in TARGET_SPHERES:
            m = ensemble_curves[s]
            sd = ensemble_sd[s]
            plt.plot(tau_common, m, linewidth=2.0, label=s.capitalize())
            plt.fill_between(tau_common, m - sd, m + sd, alpha=0.16)
        plt.axvline(LONG_LAG_START_SEC, linestyle="--", linewidth=1)
        plt.axhline(DECAY_THRESHOLD, linestyle=":", linewidth=1)
        plt.xlabel("Lag τ (s)")
        plt.ylabel("Ensemble mean normalized autocorrelation")
        plt.title("Material-level ensemble autocorrelation")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "ensemble_mean_autocorrelation.png"), dpi=180)
        plt.savefig(os.path.join(out_dir, "ensemble_mean_autocorrelation.pdf"))
        plt.close(fig)

        # 6. Patch-wise AUC profiles
        fig = plt.figure(figsize=(8.2, 5.4))
        x = np.arange(1, 10)
        for s in TARGET_SPHERES:
            means = [patch_stats[(s, p)]["long_lag_positive_auc"]["mean"] for p in PATCHES]
            errs = [patch_stats[(s, p)]["long_lag_positive_auc"]["sd"] for p in PATCHES]
            errs = [0.0 if not finite(e) else e for e in errs]
            plt.errorbar(x, means, yerr=errs, marker="o", capsize=4, label=s.capitalize())
        plt.xticks(x, PATCHES)
        plt.xlabel("Fixed image-centered patch")
        plt.ylabel("Positive ACF area (0.5-2.0 s)")
        plt.title("Patch-wise long-lag persistence (SD)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "patch_long_lag_AUC_profiles.png"), dpi=180)
        plt.savefig(os.path.join(out_dir, "patch_long_lag_AUC_profiles.pdf"))
        plt.close(fig)

        # --------------------------------------------------------
        # Human-readable summary
        # --------------------------------------------------------
        with open(os.path.join(out_dir, "comparison_summary.txt"), "w", encoding="utf-8") as f:
            f.write("Extended Paper 4 temporal-persistence comparison\n")
            f.write("================================================\n\n")
            f.write(
                f"Long-lag window: {LONG_LAG_START_SEC:.2f}-{LONG_LAG_END_SEC:.2f} s\n"
            )
            f.write(
                f"Decay definition: first lag >= 0.10 s where ACF <= {DECAY_THRESHOLD:.2f}\n\n"
            )

            for s in TARGET_SPHERES:
                f.write(f"{s.capitalize()} (n={len(by_sphere[s])} runs)\n")
                f.write("----------------------------------------\n")
                for key, label in [
                    ("mean_S", "Run mean S_peak"),
                    ("within_SD", "Within-run SD across 9 patches"),
                    ("AUC_pos", "Long-lag positive AUC"),
                    ("AUC_signed", "Long-lag signed AUC"),
                    ("decay", "Decay time to ACF<=0.10"),
                ]:
                    st = material_stats[s][key]
                    f.write(
                        f"{label}: {st['mean']:.8f} ± {st['sd']:.8f} SD\n"
                    )
                f.write("\n")

            if skipped:
                f.write("Skipped old/incompatible runs:\n")
                for d, reason in skipped:
                    f.write(f"  {os.path.basename(d)}: {reason}\n")

        logw("Saved extended temporal-shape comparison.")
        logw("DONE.")

    print("\n[INFO] Extended comparison complete.")
    print(f"[INFO] Results: {out_dir}")


if __name__ == "__main__":
    main()
