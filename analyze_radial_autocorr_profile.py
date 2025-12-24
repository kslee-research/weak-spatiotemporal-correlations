# analyze_radial_autocorr_profile.py
# Radial persistence profile using single-patch autocorrelation (SAFE VERSION)
#
# Key points:
#   - r is distance from sphere surface (NOT center).
#   - LEFT side only.
#   - Control ROI is FIXED in screen coordinates (prevents out-of-frame).
#   - If any radial ROI falls out-of-frame, it is skipped (NaN) to avoid corruption.
#
# Output:
#   results_radial_autocorr/run_YYYYmmdd_HHMMSS/
#     - run_log.txt
#     - radial_profile.csv
#     - S_vs_r.png / pdf
#     - autocorr_curves.png / pdf
#     - autocorr_map.png / pdf

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================
# 0) Matplotlib global settings (PDF-friendly)
# ============================================================
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"]  = 42
plt.rcParams["figure.dpi"]   = 150

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

CENTER_FILE = "sphere_center_radius.npy"

# Physical sphere radius in cm (fixed)
SPHERE_RADIUS_CM = 3.0

# Side configuration
SIDE = "left"   # only left supported

# Radial sampling (from surface, outward)
R0_CM   = 0.9     # start distance from surface (cm)
DR_CM   = 0.4     # step (cm)
N_R     = 15      # number of radial points

# Patch geometry (in pixels)
PATCH_W = 3       # width in x
PATCH_H = 21      # height in y (odd recommended); y ± 10 -> 21

# Time-lag configuration for autocorr
MAX_LAG_SEC = 2.0     # compute up to this lag
MIN_LAG_SEC = 0.10    # summarize ignoring too-small lag region

# Control ROI (fixed screen coordinate)
CONTROL_X_PX = 50      # px from left edge (safe)
CONTROL_DY_PX = 0      # offset in y relative to yc (0 = same row)
CONTROL_PATCH_W = PATCH_W
CONTROL_PATCH_H = PATCH_H

# Frame handling
MAX_FRAMES = None      # set int for quick test; None = all frames

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
    for name in VIDEO_CANDIDATES:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p
    return None

def load_center(center_path: str):
    arr = np.load(center_path)
    arr = np.array(arr).reshape(-1)
    if arr.size < 3:
        raise ValueError("sphere_center_radius.npy must contain at least 3 numbers: xc, yc, R_pix")
    xc, yc, R = float(arr[0]), float(arr[1]), float(arr[2])
    return xc, yc, R

def roi_from_center(xc, yc, w, h):
    half_w = w // 2
    half_h = h // 2
    x0 = int(round(xc - half_w))
    x1 = int(round(xc + half_w + 1))
    y0 = int(round(yc - half_h))
    y1 = int(round(yc + half_h + 1))
    return (x0, y0, x1, y1)

def roi_in_frame(roi, W, H, min_w=1, min_h=1):
    x0,y0,x1,y1 = roi
    if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
        return False
    # intersection size
    ix0 = max(0, x0); iy0 = max(0, y0)
    ix1 = min(W, x1); iy1 = min(H, y1)
    if (ix1 - ix0) < min_w or (iy1 - iy0) < min_h:
        return False
    return True

def clamp_roi(roi, W, H):
    x0,y0,x1,y1 = roi
    x0c = max(0, min(W-1, x0))
    y0c = max(0, min(H-1, y0))
    x1c = max(0, min(W,   x1))
    y1c = max(0, min(H,   y1))
    return (x0c,y0c,x1c,y1c)

def extract_ts_mean_gray(video_path, rois, max_frames=None):
    """
    rois: list of roi (x0,y0,x1,y1). We'll clamp inside each frame.
    returns: ts shape (n_rois, T), fps
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 29.97
    fps = float(fps)

    n_rois = len(rois)
    series = [ [] for _ in range(n_rois) ]

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
            x0,y0,x1,y1 = r
            if (x1 - x0) <= 0 or (y1 - y0) <= 0:
                series[i].append(np.nan)
            else:
                patch = gray[y0:y1, x0:x1]
                series[i].append(float(np.mean(patch)))

        t += 1

    cap.release()

    T = len(series[0])
    ts = np.zeros((n_rois, T), dtype=np.float32)
    for i in range(n_rois):
        ts[i,:] = np.array(series[i], dtype=np.float32)
    return ts, fps

def zscore_nan(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if (not np.isfinite(s)) or s < eps:
        return np.zeros_like(x)
    return (x - m) / s

def regress_residual(x, g, eps=1e-8):
    """
    residual = x - alpha*g, alpha=(x·g)/(g·g)
    x,g should be finite arrays (NaNs handled before).
    """
    num = float(np.dot(x, g))
    den = float(np.dot(g, g))
    if abs(den) < eps:
        return x.copy(), 0.0
    a = num / den
    return (x - a*g), a

def norm_autocorr(x, max_lag):
    """
    Normalized autocorrelation for lags 0..max_lag.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    var = float(np.dot(x, x))
    if var <= 1e-12:
        return np.zeros(max_lag+1, dtype=np.float64)

    N = x.size
    r = np.zeros(max_lag+1, dtype=np.float64)
    for k in range(max_lag+1):
        r[k] = float(np.dot(x[:N-k], x[k:])) / var
    return r

def fill_nan_linear(x):
    """
    Replace NaNs by linear interpolation; edges filled by nearest finite.
    If all NaN -> zeros.
    """
    x = np.asarray(x, dtype=np.float64)
    if np.all(~np.isfinite(x)):
        return np.zeros_like(x)
    idx = np.arange(x.size)
    good = np.isfinite(x)
    x2 = x.copy()
    x2[~good] = np.interp(idx[~good], idx[good], x[good])
    return x2

# ============================================================
# 3) Main
# ============================================================
def main():
    video_path = find_video(BASE_DIR)
    if video_path is None:
        raise FileNotFoundError(f"No video found in {BASE_DIR}. Expected one of: {VIDEO_CANDIDATES}")

    center_path = os.path.join(BASE_DIR, CENTER_FILE)
    if not os.path.exists(center_path):
        raise FileNotFoundError(f"Missing {CENTER_FILE} in {BASE_DIR}")

    out_root = os.path.join(BASE_DIR, "results_radial_autocorr")
    os.makedirs(out_root, exist_ok=True)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "run_log.txt")
    with open(log_path, "w", encoding="utf-8") as fp:
        log_write(fp, f"Video: {video_path}")

        xc, yc, R_pix = load_center(center_path)
        log_write(fp, f"Center: xc={xc:.2f}, yc={yc:.2f}, R_pix={R_pix:.2f}")

        pix_per_cm = R_pix / SPHERE_RADIUS_CM
        log_write(fp, f"pix_per_cm = {pix_per_cm:.4f} (R_pix / {SPHERE_RADIUS_CM}cm)")

        # We need frame size to validate ROI placement BEFORE extraction.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        ok, frame0 = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Failed to read first frame.")
        H0, W0 = frame0.shape[:2]
        log_write(fp, f"Frame size: W={W0}, H={H0}")

        # Sphere left surface x
        x_surface_left = xc - R_pix

        # Build radial ROIs
        r_list = [R0_CM + i*DR_CM for i in range(N_R)]
        radial_rois = []
        radial_meta = []  # (idx, r_cm, x_center, y_center, valid)

        for i, r_cm in enumerate(r_list):
            if SIDE != "left":
                raise ValueError("This script supports SIDE='left' only.")
            x_center = x_surface_left - (r_cm * pix_per_cm)
            y_center = yc

            roi = roi_from_center(x_center, y_center, PATCH_W, PATCH_H)
            valid = roi_in_frame(roi, W0, H0, min_w=2, min_h=2)
            radial_rois.append(roi)
            radial_meta.append((i, r_cm, x_center, y_center, valid))

        # Control ROI (fixed)
        x_ctrl = float(CONTROL_X_PX)
        y_ctrl = float(yc + CONTROL_DY_PX)
        roi_ctrl = roi_from_center(x_ctrl, y_ctrl, CONTROL_PATCH_W, CONTROL_PATCH_H)
        valid_ctrl = roi_in_frame(roi_ctrl, W0, H0, min_w=2, min_h=2)

        log_write(fp, f"Control ROI (fixed): x={x_ctrl:.1f}px, y={y_ctrl:.1f}px, valid={valid_ctrl}, roi={roi_ctrl}")

        # Log which radial ROIs are valid
        n_valid = 0
        for (i, r_cm, x_center, y_center, valid) in radial_meta:
            log_write(fp, f"r[{i:02d}]={r_cm:5.2f}cm -> x={x_center:8.1f}, y={y_center:7.1f}, valid={valid}")
            if valid:
                n_valid += 1

        if n_valid < 3:
            log_write(fp, "WARNING: Too few valid radial ROIs inside frame. Consider decreasing N_R or DR_CM, or moving camera framing.")
        if not valid_ctrl:
            raise RuntimeError("Control ROI is not valid in frame. Increase CONTROL_X_PX or adjust CONTROL_DY_PX.")

        # Extract time series for ALL ROIs (radial + control)
        rois_all = radial_rois + [roi_ctrl]
        ts, fps = extract_ts_mean_gray(video_path, rois_all, max_frames=MAX_FRAMES)
        T = ts.shape[1]
        log_write(fp, f"FPS={fps:.4f}, frames={T}, total_rois={ts.shape[0]} (radial + control)")

        # Autocorr lag setup
        max_lag = int(round(MAX_LAG_SEC * fps))
        min_lag = int(round(MIN_LAG_SEC * fps))
        max_lag = max(5, max_lag)
        min_lag = max(1, min_lag)
        if min_lag >= max_lag:
            min_lag = max(1, max_lag // 3)
        log_write(fp, f"Autocorr lags: min_lag={min_lag} ({min_lag/fps:.3f}s), max_lag={max_lag} ({max_lag/fps:.3f}s)")

        # Control signal (fill NaN then zscore)
        g_raw = ts[-1, :]
        g_raw = fill_nan_linear(g_raw)
        g = zscore_nan(g_raw)

        # For each radial ROI:
        S = np.full(N_R, np.nan, dtype=np.float64)   # persistence
        A = np.full(N_R, np.nan, dtype=np.float64)   # alpha
        ac_arr = np.full((N_R, max_lag+1), np.nan, dtype=np.float64)

        for i in range(N_R):
            valid = radial_meta[i][4]
            if not valid:
                log_write(fp, f"r[{i:02d}] skipped (out-of-frame).")
                continue

            x_raw = ts[i, :]
            x_raw = fill_nan_linear(x_raw)
            x = zscore_nan(x_raw)

            x_res, alpha = regress_residual(x, g)
            ac = norm_autocorr(x_res, max_lag=max_lag)

            A[i] = alpha
            ac_arr[i, :] = ac
            S[i] = float(np.max(ac[min_lag:max_lag+1]))

            log_write(fp, f"r[{i:02d}] r={r_list[i]:.2f}cm  alpha={alpha:+.3f}  S={S[i]:+.4f}")

        # Save CSV
        csv_path = os.path.join(out_dir, "radial_profile.csv")
        with open(csv_path, "w", encoding="utf-8") as fcsv:
            fcsv.write("idx,r_cm,x_center_px,y_center_px,valid,alpha_control,S_peak\n")
            for (i, r_cm, x_center, y_center, valid) in radial_meta:
                a = A[i] if np.isfinite(A[i]) else np.nan
                s = S[i] if np.isfinite(S[i]) else np.nan
                fcsv.write(f"{i},{r_cm:.6f},{x_center:.6f},{y_center:.6f},{int(valid)},{a:.8f},{s:.8f}\n")
        log_write(fp, f"Saved: {csv_path}")

        # Plot S vs r (valid points only)
        valid_mask = np.isfinite(S)
        r_valid = np.array(r_list)[valid_mask]
        S_valid = S[valid_mask]

        fig = plt.figure()
        if r_valid.size > 0:
            plt.plot(r_valid, S_valid, marker="o")
        plt.xlabel("Distance from sphere surface r (cm)")
        plt.ylabel("Persistence S(r) = max autocorr(residual) in lag window")
        plt.title("Radial Persistence Profile (single-patch autocorr, control-residualized)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_png = os.path.join(out_dir, "S_vs_r.png")
        out_pdf = os.path.join(out_dir, "S_vs_r.pdf")
        plt.savefig(out_png, dpi=150)
        plt.savefig(out_pdf)
        plt.close(fig)
        log_write(fp, f"Saved: {out_png}")
        log_write(fp, f"Saved: {out_pdf}")

        # Plot selected autocorr curves (near/mid/far among VALID)
        tau = np.arange(max_lag+1) / fps

        # pick indices: first valid, middle valid, last valid
        valid_indices = np.where(valid_mask)[0].tolist()
        sel = []
        if len(valid_indices) >= 1:
            sel.append(valid_indices[0])
        if len(valid_indices) >= 3:
            sel.append(valid_indices[len(valid_indices)//2])
        if len(valid_indices) >= 2:
            sel.append(valid_indices[-1])
        sel = sorted(list(set(sel)))

        fig = plt.figure()
        for j in sel:
            plt.plot(tau, ac_arr[j, :], label=f"r={r_list[j]:.2f} cm")
        plt.axvline(min_lag/fps, linestyle="--", linewidth=1)
        plt.xlabel("Lag τ (s)")
        plt.ylabel("Normalized autocorr (residual)")
        plt.title("Autocorr curves (selected radii)")
        plt.grid(True, alpha=0.3)
        if len(sel) > 0:
            plt.legend()
        plt.tight_layout()
        out_png = os.path.join(out_dir, "autocorr_curves.png")
        out_pdf = os.path.join(out_dir, "autocorr_curves.pdf")
        plt.savefig(out_png, dpi=150)
        plt.savefig(out_pdf)
        plt.close(fig)
        log_write(fp, f"Saved: {out_png}")
        log_write(fp, f"Saved: {out_pdf}")

        # Autocorr map (r × τ) using only valid rows
        if r_valid.size > 0:
            ac_valid = ac_arr[valid_mask, :]
            fig = plt.figure()
            plt.imshow(
                ac_valid,
                aspect="auto",
                origin="lower",
                extent=[0, max_lag/fps, r_valid[0], r_valid[-1]]
            )
            plt.xlabel("Lag τ (s)")
            plt.ylabel("Distance r (cm)")
            plt.title("Autocorr (residual) map: r × τ")
            plt.colorbar(label="autocorr")
            plt.tight_layout()
            out_png = os.path.join(out_dir, "autocorr_map.png")
            out_pdf = os.path.join(out_dir, "autocorr_map.pdf")
            plt.savefig(out_png, dpi=150)
            plt.savefig(out_pdf)
            plt.close(fig)
            log_write(fp, f"Saved: {out_png}")
            log_write(fp, f"Saved: {out_pdf}")
        else:
            log_write(fp, "Autocorr map skipped: no valid radial ROIs.")

        log_write(fp, "DONE.")

if __name__ == "__main__":
    main()
