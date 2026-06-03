# combine_target_distance_center_autocorr.py
# Combine Part-2 target-distance center-ROI temporal persistence results.
#
# Input:
#   results_radial_autocorr_spatial_extension/run_*/
#     - radial_profile_spatial_extension.csv
#     - center_autocorr_curve.csv
#
# Main outputs:
#   - combined_<label>_center_profile.csv
#   - combined_<label>_autocorr_curves_by_distance.png/pdf
#   - combined_<label>_persistence_duration_vs_distance.png/pdf
#   - combined_<label>_S_pos_area_vs_distance.png/pdf
#
# Interpretation:
#   The main figure is distance -> autocorr curve.
#   S_peak is retained in the CSV only as a reference, not the main metric.

import os
import csv
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 150

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(BASE_DIR, "results_radial_autocorr_spatial_extension")
COMBINED_ROOT = os.path.join(RESULTS_ROOT, "combined_results")
PROFILE_FILENAME = "radial_profile_spatial_extension.csv"
CURVE_FILENAME = "center_autocorr_curve.csv"

VALID_ONLY = True


def log_write(fp, s: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {s}"
    print(line)
    fp.write(line + "\n")
    fp.flush()


def safe_float(x):
    try:
        if x is None:
            return np.nan
        s = str(x).strip()
        if s == "":
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def safe_int(x, default=0):
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def find_run_folders(results_root):
    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"Missing folder: {results_root}")
    folders = []
    for name in sorted(os.listdir(results_root)):
        run_dir = os.path.join(results_root, name)
        if not os.path.isdir(run_dir):
            continue
        if not name.startswith("run_"):
            continue
        profile = os.path.join(run_dir, PROFILE_FILENAME)
        curve = os.path.join(run_dir, CURVE_FILENAME)
        if os.path.exists(profile) and os.path.exists(curve):
            folders.append(run_dir)
    return folders


def read_profile_csv(csv_path):
    rows = []
    run_folder = os.path.basename(os.path.dirname(csv_path))
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            valid = safe_int(row.get("valid"), default=0)
            r_global_cm = safe_float(row.get("r_global_cm"))
            if VALID_ONLY and (valid != 1 or not np.isfinite(r_global_cm)):
                continue
            rows.append({
                "run_folder": run_folder,
                "source_csv": csv_path,
                "video_number": safe_int(row.get("video_number"), default=-1),
                "frame_width_cm": safe_float(row.get("frame_width_cm")),
                "segment_start_cm": safe_float(row.get("segment_start_cm")),
                "idx": safe_int(row.get("idx"), default=0),
                "r_local_cm": safe_float(row.get("r_local_cm")),
                "r_global_cm": r_global_cm,
                "x_center_px": safe_float(row.get("x_center_px")),
                "y_center_px": safe_float(row.get("y_center_px")),
                "valid": valid,
                "S_peak": safe_float(row.get("S_peak")),
                "S_pos_area": safe_float(row.get("S_pos_area")),
                "S_signed_area": safe_float(row.get("S_signed_area")),
                "persistence_duration_sec": safe_float(row.get("persistence_duration_sec")),
                "last_positive_lag_sec": safe_float(row.get("last_positive_lag_sec")),
                "first_positive_lag_sec": safe_float(row.get("first_positive_lag_sec")),
                "mean_delayed_ac": safe_float(row.get("mean_delayed_ac")),
                "mean_positive_ac": safe_float(row.get("mean_positive_ac")),
                "persistence_threshold": safe_float(row.get("persistence_threshold")),
                "min_lag_sec": safe_float(row.get("min_lag_sec")),
                "max_lag_sec": safe_float(row.get("max_lag_sec")),
            })
    return rows


def read_curve_csv(csv_path):
    rows = []
    run_folder = os.path.basename(os.path.dirname(csv_path))
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "run_folder": run_folder,
                "video_number": safe_int(row.get("video_number"), default=-1),
                "r_local_cm": safe_float(row.get("r_local_cm")),
                "r_global_cm": safe_float(row.get("r_global_cm")),
                "tau_sec": safe_float(row.get("tau_sec")),
                "autocorr": safe_float(row.get("autocorr")),
                "min_lag_sec": safe_float(row.get("min_lag_sec")),
                "max_lag_sec": safe_float(row.get("max_lag_sec")),
                "persistence_threshold": safe_float(row.get("persistence_threshold")),
                "source_csv": csv_path,
            })
    return rows


def write_csv(rows, out_csv, fieldnames):
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    condition_label = input("Enter condition label (example: null, hollow, solid, tungsten): ").strip().lower()
    if condition_label == "":
        raise ValueError("Condition label cannot be empty.")

    print("\n=== Combine target-distance center-ROI temporal persistence profiles ===")
    print("Main output: target distance -> autocorr curve and persistence duration.\n")

    run_folders = find_run_folders(RESULTS_ROOT)
    if len(run_folders) == 0:
        raise FileNotFoundError(
            f"No run folders containing both {PROFILE_FILENAME} and {CURVE_FILENAME} found under: {RESULTS_ROOT}"
        )

    os.makedirs(COMBINED_ROOT, exist_ok=True)
    combined_id = datetime.now().strftime("combined_center_%Y%m%d_%H%M%S")
    out_dir = os.path.join(COMBINED_ROOT, combined_id)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "combine_log.txt")
    with open(log_path, "w", encoding="utf-8") as fp:
        log_write(fp, f"Results root: {RESULTS_ROOT}")
        log_write(fp, f"Found center-ROI run folders: {len(run_folders)}")
        for d in run_folders:
            log_write(fp, f"  - {d}")

        profile_rows = []
        curve_rows = []
        for d in run_folders:
            pr = read_profile_csv(os.path.join(d, PROFILE_FILENAME))
            cr = read_curve_csv(os.path.join(d, CURVE_FILENAME))
            log_write(fp, f"Read {len(pr)} profile rows and {len(cr)} curve rows from {os.path.basename(d)}")
            profile_rows.extend(pr)
            curve_rows.extend(cr)

        if len(profile_rows) == 0 or len(curve_rows) == 0:
            raise RuntimeError("No valid center-ROI rows found.")

        profile_rows = sorted(profile_rows, key=lambda r: (r["r_global_cm"], r["video_number"], r["run_folder"]))
        curve_rows = sorted(curve_rows, key=lambda r: (r["r_global_cm"], r["tau_sec"], r["run_folder"]))

        profile_csv = os.path.join(out_dir, f"combined_{condition_label}_center_profile.csv")
        profile_fields = [
            "run_folder", "video_number", "frame_width_cm", "segment_start_cm", "idx",
            "r_local_cm", "r_global_cm", "x_center_px", "y_center_px", "valid",
            "S_peak", "S_pos_area", "S_signed_area", "persistence_duration_sec",
            "last_positive_lag_sec", "first_positive_lag_sec", "mean_delayed_ac",
            "mean_positive_ac", "persistence_threshold", "min_lag_sec", "max_lag_sec", "source_csv"
        ]
        write_csv(profile_rows, profile_csv, profile_fields)
        log_write(fp, f"Saved: {profile_csv}")

        curve_csv = os.path.join(out_dir, f"combined_{condition_label}_center_autocorr_curves.csv")
        curve_fields = [
            "run_folder", "video_number", "r_local_cm", "r_global_cm", "tau_sec",
            "autocorr", "min_lag_sec", "max_lag_sec", "persistence_threshold", "source_csv"
        ]
        write_csv(curve_rows, curve_csv, curve_fields)
        log_write(fp, f"Saved: {curve_csv}")

        r = np.array([row["r_global_cm"] for row in profile_rows], dtype=float)
        dur = np.array([row["persistence_duration_sec"] for row in profile_rows], dtype=float)
        last_lag = np.array([row["last_positive_lag_sec"] for row in profile_rows], dtype=float)
        spos = np.array([row["S_pos_area"] for row in profile_rows], dtype=float)
        speak = np.array([row["S_peak"] for row in profile_rows], dtype=float)

        log_write(fp, f"Combined target-distance points: {len(profile_rows)}")
        log_write(fp, f"Global r range: {np.nanmin(r):.3f} cm to {np.nanmax(r):.3f} cm")
        log_write(fp, f"Persistence duration range: {np.nanmin(dur):+.6f} to {np.nanmax(dur):+.6f} sec")
        log_write(fp, f"S_pos_area range: {np.nanmin(spos):+.8f} to {np.nanmax(spos):+.8f}")
        log_write(fp, f"S_peak reference range: {np.nanmin(speak):+.8f} to {np.nanmax(speak):+.8f}")

        # 1) Main figure: distance -> autocorr curve
        fig = plt.figure()
        for row in profile_rows:
            rg = row["r_global_cm"]
            rf = row["run_folder"]
            cr = [c for c in curve_rows if c["run_folder"] == rf]
            tau = np.array([c["tau_sec"] for c in cr], dtype=float)
            ac = np.array([c["autocorr"] for c in cr], dtype=float)
            order = np.argsort(tau)
            plt.plot(tau[order], ac[order], linewidth=1.8, label=f"r={rg:.1f} cm")

        min_lag_vals = [row["min_lag_sec"] for row in profile_rows if np.isfinite(row["min_lag_sec"])]
        threshold_vals = [row["persistence_threshold"] for row in profile_rows if np.isfinite(row["persistence_threshold"])]
        if min_lag_vals:
            plt.axvline(float(np.nanmedian(min_lag_vals)), linestyle="--", linewidth=1, label="min delayed lag")
        if threshold_vals:
            plt.axhline(float(np.nanmedian(threshold_vals)), linestyle=":", linewidth=1, label="duration threshold")
        plt.axhline(0.0, linewidth=1)
        plt.xlabel("Lag τ (s)")
        plt.ylabel("Normalized autocorr")
        plt.title(f"Target-distance center-ROI autocorr curves: {condition_label}")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"combined_{condition_label}_autocorr_curves_by_distance.png")
        out_pdf = os.path.join(out_dir, f"combined_{condition_label}_autocorr_curves_by_distance.pdf")
        plt.savefig(out_png, dpi=150)
        plt.savefig(out_pdf)
        plt.close(fig)
        log_write(fp, f"Saved: {out_png}")
        log_write(fp, f"Saved: {out_pdf}")

        # 2) Distance -> persistence duration
        order = np.argsort(r)
        fig = plt.figure()
        plt.plot(r[order], dur[order], marker="o", linewidth=2, label="duration C(τ)>threshold")
        plt.plot(r[order], last_lag[order], marker="s", linewidth=1.5, label="last lag above threshold")
        plt.xlabel("Center-ROI distance from sphere surface r (cm)")
        plt.ylabel("Persistence time (s)")
        plt.title(f"Temporal persistence duration vs distance: {condition_label}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"combined_{condition_label}_persistence_duration_vs_distance.png")
        out_pdf = os.path.join(out_dir, f"combined_{condition_label}_persistence_duration_vs_distance.pdf")
        plt.savefig(out_png, dpi=150)
        plt.savefig(out_pdf)
        plt.close(fig)
        log_write(fp, f"Saved: {out_png}")
        log_write(fp, f"Saved: {out_pdf}")

        # 3) Distance -> positive temporal area
        fig = plt.figure()
        plt.plot(r[order], spos[order], marker="o", linewidth=2, label="S_pos_area")
        plt.xlabel("Center-ROI distance from sphere surface r (cm)")
        plt.ylabel("Integrated positive delayed persistence")
        plt.title(f"Positive delayed persistence area vs distance: {condition_label}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"combined_{condition_label}_S_pos_area_vs_distance.png")
        out_pdf = os.path.join(out_dir, f"combined_{condition_label}_S_pos_area_vs_distance.pdf")
        plt.savefig(out_png, dpi=150)
        plt.savefig(out_pdf)
        plt.close(fig)
        log_write(fp, f"Saved: {out_png}")
        log_write(fp, f"Saved: {out_pdf}")

        # 4) Reference-only S_peak plot
        fig = plt.figure()
        plt.plot(r[order], speak[order], marker="o", linewidth=1.5, label="S_peak reference")
        plt.xlabel("Center-ROI distance from sphere surface r (cm)")
        plt.ylabel("Peak delayed autocorr")
        plt.title(f"Reference S_peak vs distance: {condition_label}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(out_dir, f"combined_{condition_label}_S_peak_reference_vs_distance.png")
        out_pdf = os.path.join(out_dir, f"combined_{condition_label}_S_peak_reference_vs_distance.pdf")
        plt.savefig(out_png, dpi=150)
        plt.savefig(out_pdf)
        plt.close(fig)
        log_write(fp, f"Saved: {out_png}")
        log_write(fp, f"Saved: {out_pdf}")

        log_write(fp, "DONE.")


if __name__ == "__main__":
    main()
