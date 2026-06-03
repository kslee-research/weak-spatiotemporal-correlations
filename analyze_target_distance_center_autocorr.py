# analyze_target_distance_center_autocorr.py
# Part-2 target-distance center-ROI temporal persistence analysis
#
# Purpose:
#   Analyze ONLY the center ROI of one video.
#   The user enters the physical distance from the sphere surface to the center ROI.
#   Example target distances: 10, 30, 60, 90 cm.
#
# Main outputs:
#   - radial_profile_spatial_extension.csv   (one row per video segment)
#   - center_autocorr_curve.csv              (tau vs autocorr curve)
#   - center_autocorr_curve.png/pdf
#
# Main interpretation:
#   This version is NOT focused on S_peak. It focuses on delayed temporal persistence:
#     S_pos_area           = integral max(C(tau),0) d tau over delayed lag window
#     persistence_duration = total delayed time where C(tau) > threshold
#     last_positive_lag    = last tau in delayed window where C(tau) > threshold

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 150

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_CANDIDATES = [
    "tails_pattern.mp4",
    "tails_pattern.MP4",
    "tails_pattern.mov",
    "tails_pattern.MOV",
]

PATCH_W = 3
PATCH_H = 21

MAX_LAG_SEC = 2.0
MIN_LAG_SEC = 0.10
PERSISTENCE_THRESHOLD = 0.10

MAX_FRAMES = None


def log_write(fp, s: str):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {s}"
    print(line)
    fp.write(line + "\n")
    fp.flush()


def ask_int(prompt: str, default=None, min_value=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            val = int(default)
        else:
            try:
                val = int(raw)
            except ValueError:
                print("Please enter an integer value. Example: 1")
                continue
        if min_value is not None and val < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        return val


def ask_float(prompt: str, default=None, min_value=None):
    while True:
        raw = input(prompt).strip()
        if raw == "" and default is not None:
            val = float(default)
        else:
            try:
                val = float(raw)
            except ValueError:
                print("Please enter a numeric value. Example: 15")
                continue
        if min_value is not None and val < min_value:
            print(f"Please enter a value >= {min_value}.")
            continue
        return val


def find_video(base_dir: str):
    for name in VIDEO_CANDIDATES:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p
    return None


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

    series = [[] for _ in range(len(rois))]
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
            x0, y0, x1, y1 = clamp_roi(roi, W, H)
            if (x1 - x0) <= 0 or (y1 - y0) <= 0:
                series[i].append(np.nan)
            else:
                patch = gray[y0:y1, x0:x1]
                series[i].append(float(np.mean(patch)))
        t += 1

    cap.release()

    T = len(series[0])
    ts = np.zeros((len(rois), T), dtype=np.float32)
    for i in range(len(rois)):
        ts[i, :] = np.array(series[i], dtype=np.float32)
    return ts, fps


def fill_nan_linear(x):
    x = np.asarray(x, dtype=np.float64)
    if np.all(~np.isfinite(x)):
        return np.zeros_like(x)
    idx = np.arange(x.size)
    good = np.isfinite(x)
    x2 = x.copy()
    x2[~good] = np.interp(idx[~good], idx[good], x[good])
    return x2


def zscore_nan(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float64)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if (not np.isfinite(s)) or s < eps:
        return np.zeros_like(x)
    return (x - m) / s


def norm_autocorr(x, max_lag):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    var = float(np.dot(x, x))
    if var <= 1e-12:
        return np.zeros(max_lag + 1, dtype=np.float64)

    N = x.size
    r = np.zeros(max_lag + 1, dtype=np.float64)
    for k in range(max_lag + 1):
        r[k] = float(np.dot(x[:N-k], x[k:])) / var
    return r


def compute_persistence_metrics(ac, min_lag, max_lag, fps, threshold):
    ac_window = ac[min_lag:max_lag + 1]
    dtau = 1.0 / fps

    s_peak = float(np.max(ac_window))
    s_pos_area = float(np.sum(np.maximum(ac_window, 0.0)) * dtau)
    s_signed_area = float(np.sum(ac_window) * dtau)

    above = ac_window > threshold
    persistence_duration = float(np.sum(above) * dtau)
    if np.any(above):
        last_idx = min_lag + np.where(above)[0][-1]
        first_idx = min_lag + np.where(above)[0][0]
        last_positive_lag = float(last_idx / fps)
        first_positive_lag = float(first_idx / fps)
    else:
        last_positive_lag = 0.0
        first_positive_lag = 0.0

    mean_delayed_ac = float(np.mean(ac_window))
    mean_positive_ac = float(np.mean(np.maximum(ac_window, 0.0)))

    return {
        "S_peak": s_peak,
        "S_pos_area": s_pos_area,
        "S_signed_area": s_signed_area,
        "persistence_duration_sec": persistence_duration,
        "last_positive_lag_sec": last_positive_lag,
        "first_positive_lag_sec": first_positive_lag,
        "mean_delayed_ac": mean_delayed_ac,
        "mean_positive_ac": mean_positive_ac,
    }


def safe_cm_label(value_cm: float):
    if abs(value_cm - round(value_cm)) < 1e-9:
        return f"{int(round(value_cm))}cm"
    return f"{value_cm:.2f}cm".replace(".", "p")


def main():
    print("\n=== Part-2 target-distance center-ROI temporal persistence analysis ===")
    print("Only the CENTER ROI of the video is analyzed.")
    print("Enter the actual physical distance from the sphere surface to the CENTER ROI.")
    print("Example target distances: 10, 30, 60, 90 cm.\n")

    video_number = ask_int("Enter video index / distance-order number (example: 1, 2, 3, 4): ", min_value=1)
    target_distance_cm = ask_float("Enter center-ROI distance from sphere surface (cm, example: 10, 30, 60, 90): ", min_value=0.0)

    # In this target-distance version, the local and global distance are the same
    # because the center ROI is physically aligned to the requested marker distance.
    frame_width_cm = np.nan
    segment_start_cm = np.nan
    r_local_cm = target_distance_cm
    r_global_cm = target_distance_cm

    video_path = find_video(BASE_DIR)
    if video_path is None:
        raise FileNotFoundError(f"No video found in {BASE_DIR}. Expected one of: {VIDEO_CANDIDATES}")

    out_root = os.path.join(BASE_DIR, "results_radial_autocorr_spatial_extension")
    os.makedirs(out_root, exist_ok=True)

    run_id = datetime.now().strftime(
        f"run_%Y%m%d_%H%M%S_video{video_number:02d}_target{safe_cm_label(target_distance_cm)}"
    )
    out_dir = os.path.join(out_root, run_id)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "run_log.txt")
    with open(log_path, "w", encoding="utf-8") as fp:
        log_write(fp, f"Video: {video_path}")
        log_write(fp, f"video_number = {video_number}")
        log_write(fp, f"frame_width_cm = NaN (target-distance mode)")
        log_write(fp, f"segment_start_cm = NaN (target-distance mode)")
        log_write(fp, f"target center-ROI distance = {r_global_cm:.6f} cm")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        ok, frame0 = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError("Failed to read first frame.")

        H0, W0 = frame0.shape[:2]
        log_write(fp, f"Frame size: W={W0}, H={H0}")

        x_center = W0 / 2.0
        y_center = H0 / 2.0
        roi = roi_from_center(x_center, y_center, PATCH_W, PATCH_H)
        valid = roi_in_frame(roi, W0, H0, min_w=2, min_h=2)
        log_write(fp, f"Center ROI: x={x_center:.2f}, y={y_center:.2f}, roi={roi}, valid={valid}")
        if not valid:
            raise RuntimeError("Center ROI is not valid inside the frame.")

        ts, fps = extract_ts_mean_gray(video_path, [roi], max_frames=MAX_FRAMES)
        T = ts.shape[1]
        log_write(fp, f"FPS={fps:.4f}, frames={T}, total_rois={ts.shape[0]}")

        max_lag = int(round(MAX_LAG_SEC * fps))
        min_lag = int(round(MIN_LAG_SEC * fps))
        max_lag = max(5, max_lag)
        min_lag = max(1, min_lag)
        if min_lag >= max_lag:
            min_lag = max(1, max_lag // 3)

        log_write(
            fp,
            f"Autocorr lags: min_lag={min_lag} ({min_lag/fps:.3f}s), "
            f"max_lag={max_lag} ({max_lag/fps:.3f}s), threshold={PERSISTENCE_THRESHOLD:.3f}"
        )

        x_raw = fill_nan_linear(ts[0, :])
        x = zscore_nan(x_raw)
        ac = norm_autocorr(x, max_lag=max_lag)
        metrics = compute_persistence_metrics(ac, min_lag, max_lag, fps, PERSISTENCE_THRESHOLD)

        for k, v in metrics.items():
            log_write(fp, f"{k} = {v:+.8f}")

        csv_path = os.path.join(out_dir, "radial_profile_spatial_extension.csv")
        with open(csv_path, "w", encoding="utf-8") as fcsv:
            fcsv.write(
                "video_number,frame_width_cm,segment_start_cm,idx,"
                "r_local_cm,r_global_cm,x_center_px,y_center_px,valid,"
                "alpha_control,S_peak,S_pos_area,S_signed_area,S_mean_pos,S_mean_signed,"
                "persistence_duration_sec,last_positive_lag_sec,first_positive_lag_sec,"
                "mean_delayed_ac,mean_positive_ac,persistence_threshold,"
                "min_lag_sec,max_lag_sec\n"
            )
            fcsv.write(
                f"{video_number},{frame_width_cm:.6f},{segment_start_cm:.6f},0,"
                f"{r_local_cm:.6f},{r_global_cm:.6f},{x_center:.6f},{y_center:.6f},1,"
                f"{np.nan:.8f},{metrics['S_peak']:.8f},{metrics['S_pos_area']:.8f},"
                f"{metrics['S_signed_area']:.8f},{metrics['mean_positive_ac']:.8f},{metrics['mean_delayed_ac']:.8f},"
                f"{metrics['persistence_duration_sec']:.8f},{metrics['last_positive_lag_sec']:.8f},"
                f"{metrics['first_positive_lag_sec']:.8f},{metrics['mean_delayed_ac']:.8f},"
                f"{metrics['mean_positive_ac']:.8f},{PERSISTENCE_THRESHOLD:.8f},"
                f"{min_lag/fps:.8f},{max_lag/fps:.8f}\n"
            )
        log_write(fp, f"Saved: {csv_path}")

        tau = np.arange(max_lag + 1) / fps
        curve_csv = os.path.join(out_dir, "center_autocorr_curve.csv")
        with open(curve_csv, "w", encoding="utf-8") as fcsv:
            fcsv.write("video_number,r_local_cm,r_global_cm,tau_sec,autocorr,min_lag_sec,max_lag_sec,persistence_threshold\n")
            for tsec, aval in zip(tau, ac):
                fcsv.write(
                    f"{video_number},{r_local_cm:.8f},{r_global_cm:.8f},{tsec:.8f},{aval:.10f},"
                    f"{min_lag/fps:.8f},{max_lag/fps:.8f},{PERSISTENCE_THRESHOLD:.8f}\n"
                )
        log_write(fp, f"Saved: {curve_csv}")

        fig = plt.figure()
        plt.plot(tau, ac, linewidth=2, label=f"r={r_global_cm:.1f} cm")
        plt.axvspan(min_lag / fps, max_lag / fps, alpha=0.12, label="delayed window")
        plt.axhline(0.0, linewidth=1)
        plt.axhline(PERSISTENCE_THRESHOLD, linestyle="--", linewidth=1, label=f"threshold={PERSISTENCE_THRESHOLD:.2f}")
        plt.xlabel("Lag τ (s)")
        plt.ylabel("Normalized autocorr")
        plt.title(f"Center-ROI temporal persistence, r={r_global_cm:.1f} cm")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(out_dir, "center_autocorr_curve.png")
        out_pdf = os.path.join(out_dir, "center_autocorr_curve.pdf")
        plt.savefig(out_png, dpi=150)
        plt.savefig(out_pdf)
        plt.close(fig)
        log_write(fp, f"Saved: {out_png}")
        log_write(fp, f"Saved: {out_pdf}")

        log_write(fp, "DONE.")


if __name__ == "__main__":
    main()
