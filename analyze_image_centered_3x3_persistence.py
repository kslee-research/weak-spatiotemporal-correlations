# analyze_image_centered_3x3_persistence_v2.py
# Paper 4: image-centered 3x3 spatial temporal-persistence analysis
#
# Purpose
#   - Preserve the existing temporal-persistence pipeline:
#       patch mean intensity -> z-score -> control residualization
#       -> normalized autocorrelation -> S_peak
#   - Remove sphere-boundary / sphere-center / radius calibration completely.
#   - Use the VIDEO FRAME CENTER as the common spatial reference.
#   - Analyze a fixed 3x3 grid (9 patches) at identical image-relative coordinates.
#
# Main outputs
#   results_center_referenced_radial_autocorr/<sphere>_YYYYmmdd_HHMMSS/
#     - run_log.txt
#     - run_metadata.json
#     - spatial_profile.csv
#     - grid_preview.png / pdf
#     - persistence_map.png / pdf
#     - autocorr_curves.png / pdf
#     - autocorr_map.png / pdf

import os
import re
import json
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
# 0) Matplotlib global settings
# ============================================================
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 150

# ============================================================
# 1) Basic configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_CANDIDATES = [
    "tails_pattern.mp4",
    "tails_pattern.MP4",
    "tails_pattern.mov",
    "tails_pattern.MOV",
]

RESULT_ROOT = "results_center_referenced_radial_autocorr"

# ------------------------------------------------------------
# 3x3 image-centered sampling geometry
# ------------------------------------------------------------
# P5 is exactly at the frame center.
# Columns are shifted by +/- GRID_DX_FRAC * frame width.
# Rows are shifted by +/- GRID_DY_FRAC * frame height.
#
# Initial defaults chosen to cover a broad central field while avoiding
# the extreme frame edges. Adjust only if the preview shows overlap with
# the shield/object in a particular acquisition layout.
GRID_DX_FRAC = 0.16
GRID_DY_FRAC = 0.22

# Same patch concept as the previous persistence analysis.
PATCH_W = 3
PATCH_H = 21

# Autocorrelation lag window: preserved from the previous analysis.
MAX_LAG_SEC = 2.0
MIN_LAG_SEC = 0.10

# Additional temporal-shape metrics for Paper 4
LONG_LAG_START_SEC = 0.50
LONG_LAG_END_SEC = 2.00
DECAY_THRESHOLD = 0.10  # first lag where autocorr falls to <= 0.10

# Fixed screen-coordinate control ROI, retained from the previous pipeline.
CONTROL_X_PX = 50
CONTROL_DY_PX = 0
CONTROL_PATCH_W = PATCH_W
CONTROL_PATCH_H = PATCH_H

# Frame handling
MAX_FRAMES = None  # int for quick test; None = all frames

# ============================================================
# 2) Helpers
# ============================================================
def log_write(fp, s: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {s}"
    print(line)
    fp.write(line + "\n")
    fp.flush()


def find_video(base_dir: str):
    # First preserve the old fixed-name behavior.
    for name in VIDEO_CANDIDATES:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p

    # Convenience fallback: if there is exactly one MOV/MP4 in the folder,
    # use it automatically.
    vids = []
    for name in os.listdir(base_dir):
        if os.path.splitext(name)[1].lower() in (".mov", ".mp4"):
            vids.append(os.path.join(base_dir, name))

    vids = sorted(vids)
    if len(vids) == 1:
        return vids[0]
    if len(vids) > 1:
        names = "\n  ".join(os.path.basename(v) for v in vids)
        raise RuntimeError(
            "Multiple video files were found. Keep only one analysis video "
            "in this folder, or rename the intended file to tails_pattern.mov/mp4.\n"
            f"  {names}"
        )
    return None


def canonical_sphere_type(raw: str):
    s = raw.strip().lower()
    aliases = {
        "s": "steel",
        "steel": "steel",
        "stainless": "steel",
        "stainless steel": "steel",
        "ss": "steel",
        "t": "tungsten",
        "w": "tungsten",
        "tungsten": "tungsten",
        "wolfram": "tungsten",
    }
    return aliases.get(s, s)


def safe_slug(text: str):
    text = text.strip().lower().replace(" ", "_")
    text = re.sub(r"[^a-z0-9_\-]+", "", text)
    return text or "unknown"


def prompt_positive_float(prompt_text: str, allow_blank=False):
    while True:
        raw = input(prompt_text).strip()
        if allow_blank and raw == "":
            return None
        try:
            value = float(raw)
            if value <= 0:
                raise ValueError
            return value
        except ValueError:
            print("[ERROR] Please enter a positive number.")


def collect_run_metadata():
    print("\n=== Paper 4 image-centered 3x3 metadata ===")
    sphere_type = canonical_sphere_type(
        input("[INPUT] Sphere type [steel/tungsten]: ")
    )
    sphere_type = safe_slug(sphere_type)

    # Kept for documentation only; not used for spatial calibration.
    diameter_cm = prompt_positive_float(
        "[INPUT] Sphere diameter in cm (optional; press Enter to skip): ",
        allow_blank=True,
    )
    mass_kg = prompt_positive_float(
        "[INPUT] Measured sphere mass in kg (optional; press Enter to skip): ",
        allow_blank=True,
    )

    return sphere_type, diameter_cm, mass_kg


def roi_from_center(xc, yc, w, h):
    half_w = w // 2
    half_h = h // 2
    x0 = int(round(xc - half_w))
    x1 = int(round(xc + half_w + 1))
    y0 = int(round(yc - half_h))
    y1 = int(round(yc + half_h + 1))
    return (x0, y0, x1, y1)


def roi_in_frame(roi, W, H, min_w=1, min_h=1):
    x0, y0, x1, y1 = roi
    if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
        return False
    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(W, x1)
    iy1 = min(H, y1)
    return (ix1 - ix0) >= min_w and (iy1 - iy0) >= min_h


def clamp_roi(roi, W, H):
    x0, y0, x1, y1 = roi
    x0c = max(0, min(W - 1, x0))
    y0c = max(0, min(H - 1, y0))
    x1c = max(0, min(W, x1))
    y1c = max(0, min(H, y1))
    return (x0c, y0c, x1c, y1c)


def extract_ts_mean_gray(video_path, rois, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 29.97
    fps = float(fps)

    n_rois = len(rois)
    series = [[] for _ in range(n_rois)]

    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames is not None and t >= max_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        H, W = gray.shape[:2]

        for i, roi in enumerate(rois):
            r = clamp_roi(roi, W, H)
            x0, y0, x1, y1 = r
            if (x1 - x0) <= 0 or (y1 - y0) <= 0:
                series[i].append(np.nan)
            else:
                patch = gray[y0:y1, x0:x1]
                series[i].append(float(np.mean(patch)))
        t += 1

    cap.release()

    if not series or len(series[0]) == 0:
        raise RuntimeError("No frames were extracted from video.")

    T = len(series[0])
    ts = np.zeros((n_rois, T), dtype=np.float32)
    for i in range(n_rois):
        ts[i, :] = np.array(series[i], dtype=np.float32)
    return ts, fps


# ----------------------------------------------------------------
# The following temporal-persistence functions are intentionally
# retained from the previous analysis.
# ----------------------------------------------------------------
def zscore_nan(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if (not np.isfinite(s)) or s < eps:
        return np.zeros_like(x)
    return (x - m) / s


def regress_residual(x, g, eps=1e-8):
    num = float(np.dot(x, g))
    den = float(np.dot(g, g))
    if abs(den) < eps:
        return x.copy(), 0.0
    alpha = num / den
    return (x - alpha * g), alpha


def norm_autocorr(x, max_lag):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    var = float(np.dot(x, x))
    if var <= 1e-12:
        return np.zeros(max_lag + 1, dtype=np.float64)

    N = x.size
    r = np.zeros(max_lag + 1, dtype=np.float64)
    for k in range(max_lag + 1):
        r[k] = float(np.dot(x[: N - k], x[k:])) / var
    return r


def fill_nan_linear(x):
    x = np.asarray(x, dtype=np.float64)
    if np.all(~np.isfinite(x)):
        return np.zeros_like(x)
    idx = np.arange(x.size)
    good = np.isfinite(x)
    x2 = x.copy()
    x2[~good] = np.interp(idx[~good], idx[good], x[good])
    return x2



def temporal_shape_metrics(ac, fps):
    """
    Calculate additional temporal-shape metrics from one normalized ACF.

    Metrics
    -------
    long_lag_positive_auc:
        Integral of max(ACF, 0) from LONG_LAG_START_SEC to LONG_LAG_END_SEC.
        This quantifies persistent positive correlation at long lag.
    long_lag_signed_auc:
        Signed ACF integral over the same window.
    decay_time_to_threshold_sec:
        First lag >= MIN_LAG_SEC where ACF <= DECAY_THRESHOLD.
        NaN if the threshold is not reached within MAX_LAG_SEC.
    """
    ac = np.asarray(ac, dtype=np.float64)
    tau = np.arange(ac.size, dtype=np.float64) / float(fps)

    mask = (tau >= LONG_LAG_START_SEC) & (tau <= LONG_LAG_END_SEC)
    if np.sum(mask) >= 2:
        t = tau[mask]
        a = ac[mask]
        positive_auc = float(np.trapz(np.maximum(a, 0.0), t))
        signed_auc = float(np.trapz(a, t))
    else:
        positive_auc = np.nan
        signed_auc = np.nan

    search = np.where((tau >= MIN_LAG_SEC) & (ac <= DECAY_THRESHOLD))[0]
    decay_time = float(tau[search[0]]) if search.size > 0 else np.nan

    return positive_auc, signed_auc, decay_time


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def build_grid(frame_w, frame_h):
    """Return metadata for a fixed 3x3 grid centered on the image center."""
    cx = frame_w / 2.0
    cy = frame_h / 2.0
    dx = GRID_DX_FRAC * frame_w
    dy = GRID_DY_FRAC * frame_h

    x_offsets = (-dx, 0.0, +dx)
    y_offsets = (-dy, 0.0, +dy)

    patches = []
    patch_num = 1
    for row, oy in enumerate(y_offsets):
        for col, ox in enumerate(x_offsets):
            x = cx + ox
            y = cy + oy
            roi = roi_from_center(x, y, PATCH_W, PATCH_H)
            valid = roi_in_frame(roi, frame_w, frame_h, min_w=2, min_h=2)
            patches.append({
                "patch_id": f"P{patch_num}",
                "patch_index": patch_num - 1,
                "row": row,
                "col": col,
                "x_center_px": float(x),
                "y_center_px": float(y),
                "x_offset_px": float(ox),
                "y_offset_px": float(oy),
                "x_offset_frac": float(ox / frame_w),
                "y_offset_frac": float(oy / frame_h),
                "roi": roi,
                "valid": bool(valid),
            })
            patch_num += 1

    return cx, cy, dx, dy, patches


def save_grid_preview(frame_bgr, patches, cx, cy, out_dir):
    img = frame_bgr.copy()

    # frame center
    cv2.drawMarker(
        img,
        (int(round(cx)), int(round(cy))),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
    )

    for p in patches:
        x0, y0, x1, y1 = p["roi"]
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 2)
        cv2.putText(
            img,
            p["patch_id"],
            (x0 + 5, max(20, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    png_path = os.path.join(out_dir, "grid_preview.png")
    cv2.imwrite(png_path, img)

    # PDF version via matplotlib
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(10, 6))
    plt.imshow(rgb)
    plt.axis("off")
    plt.title("Image-centered 3x3 patch grid")
    plt.tight_layout()
    pdf_path = os.path.join(out_dir, "grid_preview.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


# ============================================================
# 3) Main
# ============================================================
def main():
    video_path = find_video(BASE_DIR)
    if video_path is None:
        raise FileNotFoundError(
            f"No MOV/MP4 video found in {BASE_DIR}."
        )

    sphere_type, sphere_diameter_cm, sphere_mass_kg = collect_run_metadata()

    out_root = os.path.join(BASE_DIR, RESULT_ROOT)
    os.makedirs(out_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{sphere_type}_{timestamp}"
    out_dir = os.path.join(out_root, run_id)
    os.makedirs(out_dir, exist_ok=False)

    log_path = os.path.join(out_dir, "run_log.txt")
    with open(log_path, "w", encoding="utf-8") as fp:
        log_write(fp, f"Run ID: {run_id}")
        log_write(fp, f"Video: {video_path}")
        log_write(fp, f"Sphere type: {sphere_type}")
        if sphere_diameter_cm is None:
            log_write(fp, "Sphere diameter: not entered")
        else:
            log_write(fp, f"Sphere diameter: {sphere_diameter_cm:.6f} cm")
        if sphere_mass_kg is None:
            log_write(fp, "Sphere mass: not entered")
        else:
            log_write(fp, f"Sphere mass: {sphere_mass_kg:.6f} kg")

        # Read first frame and define the grid directly from the image geometry.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        ok, frame0 = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Failed to read first frame.")

        H0, W0 = frame0.shape[:2]
        log_write(fp, f"Frame size: W={W0}, H={H0}")

        cx, cy, dx, dy, patches = build_grid(W0, H0)
        log_write(fp, f"Frame center: x={cx:.3f}, y={cy:.3f}")
        log_write(fp, f"3x3 spacing: dx={dx:.3f}px ({GRID_DX_FRAC:.4f}W), "
                      f"dy={dy:.3f}px ({GRID_DY_FRAC:.4f}H)")

        for p in patches:
            log_write(
                fp,
                f"{p['patch_id']}: row={p['row']} col={p['col']} "
                f"x={p['x_center_px']:.2f} y={p['y_center_px']:.2f} "
                f"valid={p['valid']} roi={p['roi']}"
            )

        if not all(p["valid"] for p in patches):
            raise RuntimeError(
                "One or more 3x3 patches fall outside the frame. "
                "Adjust GRID_DX_FRAC / GRID_DY_FRAC."
            )

        preview_png, preview_pdf = save_grid_preview(
            frame0, patches, cx, cy, out_dir
        )
        log_write(fp, f"Saved: {preview_png}")
        log_write(fp, f"Saved: {preview_pdf}")

        # Control ROI
        x_ctrl = float(CONTROL_X_PX)
        y_ctrl = float(cy + CONTROL_DY_PX)
        roi_ctrl = roi_from_center(
            x_ctrl, y_ctrl, CONTROL_PATCH_W, CONTROL_PATCH_H
        )
        valid_ctrl = roi_in_frame(roi_ctrl, W0, H0, min_w=2, min_h=2)
        log_write(
            fp,
            f"Control ROI: x={x_ctrl:.1f}px, y={y_ctrl:.1f}px, "
            f"valid={valid_ctrl}, roi={roi_ctrl}"
        )
        if not valid_ctrl:
            raise RuntimeError(
                "Control ROI is not valid in frame. Adjust CONTROL_X_PX or CONTROL_DY_PX."
            )

        # Extract time series for 9 patches + control.
        rois_all = [p["roi"] for p in patches] + [roi_ctrl]
        ts, fps = extract_ts_mean_gray(
            video_path, rois_all, max_frames=MAX_FRAMES
        )
        T = ts.shape[1]
        log_write(
            fp,
            f"FPS={fps:.6f}, frames={T}, total_rois={ts.shape[0]} (9 patches + control)"
        )

        max_lag = int(round(MAX_LAG_SEC * fps))
        min_lag = int(round(MIN_LAG_SEC * fps))
        max_lag = max(5, max_lag)
        min_lag = max(1, min_lag)
        if min_lag >= max_lag:
            min_lag = max(1, max_lag // 3)

        log_write(
            fp,
            f"Autocorr lags: min_lag={min_lag} ({min_lag/fps:.3f}s), "
            f"max_lag={max_lag} ({max_lag/fps:.3f}s)"
        )

        # Control signal
        g_raw = fill_nan_linear(ts[-1, :])
        g = zscore_nan(g_raw)

        n_patch = len(patches)
        S = np.full(n_patch, np.nan, dtype=np.float64)
        A = np.full(n_patch, np.nan, dtype=np.float64)
        ac_arr = np.full((n_patch, max_lag + 1), np.nan, dtype=np.float64)
        long_auc_pos = np.full(n_patch, np.nan, dtype=np.float64)
        long_auc_signed = np.full(n_patch, np.nan, dtype=np.float64)
        decay_time = np.full(n_patch, np.nan, dtype=np.float64)

        for i, p in enumerate(patches):
            x_raw = fill_nan_linear(ts[i, :])
            x = zscore_nan(x_raw)

            x_res, alpha = regress_residual(x, g)
            ac = norm_autocorr(x_res, max_lag=max_lag)

            A[i] = alpha
            ac_arr[i, :] = ac
            S[i] = float(np.max(ac[min_lag:max_lag + 1]))
            (long_auc_pos[i], long_auc_signed[i], decay_time[i]) = temporal_shape_metrics(ac, fps)

            log_write(
                fp,
                f"{p['patch_id']}: alpha={alpha:+.6f} S_peak={S[i]:+.6f} "
                f"AUC+={long_auc_pos[i]:+.6f} AUC_signed={long_auc_signed[i]:+.6f} "
                f"decay_t={decay_time[i]:.4f}s"
            )

        # Save metadata
        metadata = {
            "analysis": "Paper 4 image-centered 3x3 temporal persistence",
            "run_id": run_id,
            "timestamp": timestamp,
            "sphere_type": sphere_type,
            "sphere_diameter_cm": sphere_diameter_cm,
            "sphere_mass_kg": sphere_mass_kg,
            "frame_width_px": W0,
            "frame_height_px": H0,
            "frame_center_x_px": cx,
            "frame_center_y_px": cy,
            "grid_dx_frac": GRID_DX_FRAC,
            "grid_dy_frac": GRID_DY_FRAC,
            "grid_dx_px": dx,
            "grid_dy_px": dy,
            "patch_w_px": PATCH_W,
            "patch_h_px": PATCH_H,
            "control_x_px": CONTROL_X_PX,
            "control_dy_px": CONTROL_DY_PX,
            "fps": fps,
            "frames": T,
            "min_lag_sec": MIN_LAG_SEC,
            "max_lag_sec": MAX_LAG_SEC,
            "long_lag_start_sec": LONG_LAG_START_SEC,
            "long_lag_end_sec": LONG_LAG_END_SEC,
            "decay_threshold": DECAY_THRESHOLD,
            "source_video": os.path.basename(video_path),
            "patches": [
                {
                    k: v for k, v in p.items()
                    if k != "roi"
                } | {"roi": list(p["roi"])}
                for p in patches
            ],
        }
        metadata_path = os.path.join(out_dir, "run_metadata.json")
        write_json(metadata_path, metadata)
        log_write(fp, f"Saved: {metadata_path}")

        # Save all 9 patch values.
        csv_path = os.path.join(out_dir, "spatial_profile.csv")
        fieldnames = [
            "run_id", "sphere_type", "sphere_diameter_cm", "sphere_mass_kg",
            "patch_id", "patch_index", "row", "col",
            "x_center_px", "y_center_px",
            "x_offset_px", "y_offset_px",
            "x_offset_frac", "y_offset_frac",
            "alpha_control", "S_peak",
            "long_lag_positive_auc", "long_lag_signed_auc",
            "decay_time_to_0p1_sec", "fps", "frames"
        ]
        with open(csv_path, "w", encoding="utf-8", newline="") as fcsv:
            writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
            writer.writeheader()
            for i, p in enumerate(patches):
                writer.writerow({
                    "run_id": run_id,
                    "sphere_type": sphere_type,
                    "sphere_diameter_cm": "" if sphere_diameter_cm is None else f"{sphere_diameter_cm:.8f}",
                    "sphere_mass_kg": "" if sphere_mass_kg is None else f"{sphere_mass_kg:.8f}",
                    "patch_id": p["patch_id"],
                    "patch_index": p["patch_index"],
                    "row": p["row"],
                    "col": p["col"],
                    "x_center_px": f"{p['x_center_px']:.8f}",
                    "y_center_px": f"{p['y_center_px']:.8f}",
                    "x_offset_px": f"{p['x_offset_px']:.8f}",
                    "y_offset_px": f"{p['y_offset_px']:.8f}",
                    "x_offset_frac": f"{p['x_offset_frac']:.8f}",
                    "y_offset_frac": f"{p['y_offset_frac']:.8f}",
                    "alpha_control": f"{A[i]:.8f}",
                    "S_peak": f"{S[i]:.8f}",
                    "long_lag_positive_auc": f"{long_auc_pos[i]:.8f}",
                    "long_lag_signed_auc": f"{long_auc_signed[i]:.8f}",
                    "decay_time_to_0p1_sec": "" if not np.isfinite(decay_time[i]) else f"{decay_time[i]:.8f}",
                    "fps": f"{fps:.8f}",
                    "frames": T,
                })
        log_write(fp, f"Saved: {csv_path}")

        # Save the FULL autocorrelation arrays numerically so later comparison
        # can calculate ensemble mean curves, long-lag AUC, and decay-time metrics
        # without extracting values from figures.
        ac_csv_path = os.path.join(out_dir, "autocorr_curves.csv")
        tau = np.arange(max_lag + 1, dtype=np.float64) / fps
        with open(ac_csv_path, "w", encoding="utf-8", newline="") as fac:
            fieldnames_ac = ["lag_index", "lag_sec"] + [f"P{i}" for i in range(1, 10)]
            writer = csv.DictWriter(fac, fieldnames=fieldnames_ac)
            writer.writeheader()
            for k in range(max_lag + 1):
                row = {
                    "lag_index": k,
                    "lag_sec": f"{tau[k]:.8f}",
                }
                for i in range(9):
                    row[f"P{i+1}"] = f"{ac_arr[i, k]:.8f}"
                writer.writerow(row)
        log_write(fp, f"Saved: {ac_csv_path}")

        # Compact per-patch temporal-shape table
        metric_csv_path = os.path.join(out_dir, "temporal_metrics.csv")
        with open(metric_csv_path, "w", encoding="utf-8", newline="") as fmt:
            fieldnames_tm = [
                "run_id", "sphere_type", "patch_id", "S_peak",
                "long_lag_positive_auc", "long_lag_signed_auc",
                "decay_time_to_0p1_sec"
            ]
            writer = csv.DictWriter(fmt, fieldnames=fieldnames_tm)
            writer.writeheader()
            for i, p in enumerate(patches):
                writer.writerow({
                    "run_id": run_id,
                    "sphere_type": sphere_type,
                    "patch_id": p["patch_id"],
                    "S_peak": f"{S[i]:.8f}",
                    "long_lag_positive_auc": f"{long_auc_pos[i]:.8f}",
                    "long_lag_signed_auc": f"{long_auc_signed[i]:.8f}",
                    "decay_time_to_0p1_sec": "" if not np.isfinite(decay_time[i]) else f"{decay_time[i]:.8f}",
                })
        log_write(fp, f"Saved: {metric_csv_path}")

        # --------------------------------------------------------
        # Figure 1: 3x3 persistence map
        # --------------------------------------------------------
        S_map = S.reshape(3, 3)
        fig = plt.figure(figsize=(6.2, 5.4))
        im = plt.imshow(S_map, origin="upper", aspect="equal")
        plt.colorbar(im, label="Persistence S_peak")
        plt.xticks([0, 1, 2], ["Left", "Center", "Right"])
        plt.yticks([0, 1, 2], ["Top", "Center", "Bottom"])
        plt.title(f"Image-centered 3x3 persistence map: {sphere_type}")

        for r in range(3):
            for c in range(3):
                pid = f"P{r*3+c+1}"
                plt.text(
                    c, r, f"{pid}\n{S_map[r, c]:.3f}",
                    ha="center", va="center"
                )

        plt.tight_layout()
        out_png = os.path.join(out_dir, "persistence_map.png")
        out_pdf = os.path.join(out_dir, "persistence_map.pdf")
        plt.savefig(out_png, dpi=180)
        plt.savefig(out_pdf)
        plt.close(fig)
        log_write(fp, f"Saved: {out_png}")
        log_write(fp, f"Saved: {out_pdf}")

        # --------------------------------------------------------
        # Figure 2: all 9 autocorrelation curves on one page
        # --------------------------------------------------------
        tau = np.arange(max_lag + 1) / fps
        fig = plt.figure(figsize=(8.0, 5.8))
        for i, p in enumerate(patches):
            plt.plot(tau, ac_arr[i, :], linewidth=1.1, label=p["patch_id"])
        plt.axvline(min_lag / fps, linestyle="--", linewidth=1)
        plt.xlabel("Lag τ (s)")
        plt.ylabel("Normalized autocorr (residual)")
        plt.title(f"Autocorrelation curves for 9 fixed patches: {sphere_type}")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=3, fontsize=9)
        plt.tight_layout()
        out_png = os.path.join(out_dir, "autocorr_curves.png")
        out_pdf = os.path.join(out_dir, "autocorr_curves.pdf")
        plt.savefig(out_png, dpi=180)
        plt.savefig(out_pdf)
        plt.close(fig)
        log_write(fp, f"Saved: {out_png}")
        log_write(fp, f"Saved: {out_pdf}")

        # --------------------------------------------------------
        # Figure 3: patch index x lag autocorrelation map
        # --------------------------------------------------------
        fig = plt.figure(figsize=(8.0, 5.6))
        im = plt.imshow(
            ac_arr,
            aspect="auto",
            origin="lower",
            extent=[0, max_lag / fps, 0.5, 9.5],
        )
        plt.xlabel("Lag τ (s)")
        plt.ylabel("Patch")
        plt.yticks(range(1, 10), [f"P{i}" for i in range(1, 10)])
        plt.title(f"Autocorrelation map: 9 fixed patches × lag ({sphere_type})")
        plt.colorbar(im, label="autocorr")
        plt.tight_layout()
        out_png = os.path.join(out_dir, "autocorr_map.png")
        out_pdf = os.path.join(out_dir, "autocorr_map.pdf")
        plt.savefig(out_png, dpi=180)
        plt.savefig(out_pdf)
        plt.close(fig)
        log_write(fp, f"Saved: {out_png}")
        log_write(fp, f"Saved: {out_pdf}")

        log_write(fp, "DONE.")

    print("\n[INFO] Image-centered 3x3 persistence analysis complete.")
    print(f"[INFO] Output folder: {out_dir}")
    print("[INFO] Use spatial_profile.csv for the later multi-run comparison.")


if __name__ == "__main__":
    main()
