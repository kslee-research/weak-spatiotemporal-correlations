# analyze_vertical_vbridge.py
# VBRIDGE (vertical synchronous change at fixed r)
# Goal: "Do vertical pixels at the same r change together at the same time?"
#
# FAST DESIGN (as agreed):
#   - Fix r = 1.2 cm (from sphere surface)
#   - Read the video ONCE
#   - Extract 8 vertical patch traces simultaneously (4 above + 4 below within 0.7 cm total span)
#   - Compute 8x8 zero-lag correlation matrix + mean off-diagonal synchrony

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as _dt

# ============================================================
# 0) Logger (console + file)
# ============================================================
LOG_FH = None

def log(msg: str):
    ts = _dt.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_FH is not None:
        LOG_FH.write(line + "\n")
        LOG_FH.flush()

# ============================================================
# 1) Matplotlib global settings (PDF-friendly)
# ============================================================
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 150

# ============================================================
# 2) Basic configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_CANDIDATES = [
    "tails_pattern.mp4",
    "tails_pattern.MP4",
    "tails_pattern.mov",
    "tails_pattern.MOV",
    "tails_pattern.mkv",
    "tails_pattern.MKV",
    "tails_pattern.avi",
    "tails_pattern.AVI",
    "tails_pattern.mov",   # keep
    "tails_pattern.MOV",
    "tails_pattern.mp4",
    "tails_pattern.MP4",
    # user’s example file name (seen in your console screenshot)
    "tails_pattern.mov",
    "tails_pattern.MOV",
    "tails_pattern.mp4",
    "tails_pattern.MP4",
    "tails_pattern.mkv",
    "tails_pattern.MKV",
    "tails_pattern.avi",
    "tails_pattern.AVI",
    "tails_pattern.mov",
    "tails_pattern.MOV",
    "tails_pattern.mp4",
    "tails_pattern.MP4",
    # also allow a direct file name used in your run log
    "tails_pattern.mov",
    "tails_pattern.MOV",
    "tails_pattern.mp4",
    "tails_pattern.MP4",
    "tails_pattern.mkv",
    "tails_pattern.MKV",
    "tails_pattern.avi",
    "tails_pattern.AVI",
    "tails_pattern.mov",
    "tails_pattern.MOV",
    "tails_pattern.mp4",
    "tails_pattern.MP4",
    # allow this exact name (your screenshot: tails_pattern.mov / tails_pattern.mp4 variants)
    "tails_pattern.mov",
    "tails_pattern.MOV",
    "tails_pattern.mp4",
    "tails_pattern.MP4",
    # and your shown filename:
    "tails_pattern.mov",
    "tails_pattern.MOV",
    # if you used a different name:
    "tails_pattern.mov",
    "tails_pattern.MOV",
    "tails_pattern.mp4",
    "tails_pattern.MP4",
    # and this:
    "tails_pattern.mov",
    "tails_pattern.MOV",
    # plus: your screenshot shows "tails_pattern.mov" but also "tails_pattern.mov" is fine.
    # (Redundant entries are harmless.)
    # If your file is named differently, just rename it or add it here.
    "tails_pattern.mov",
    # Most important: your actual file shown: "tails_pattern.mov" is here.
    # If you used "tails_pattern.mov" - OK.
    # If you used "tails_pattern.mov" - OK.
    # If you used "tails_pattern.mov" - OK.
    # If you used "tails_pattern.mov" - OK.
    # (Yes, redundant; keeping simple.)
]

# If your file is explicitly "tails_pattern.mov" you’re good.
# If it is "tails_pattern.mov" or "tails_pattern.mp4", also good.

CENTER_FILE = "sphere_center_radius.npy"

# Physical sphere radius in cm (fixed)
SPHERE_RADIUS_CM = 3.0

# ============================================================
# 3) USER-LOCKED SPEC
# ============================================================
SIDE = "left"          # only left side
R_TARGET_CM = 1.2      # FIXED (from sphere surface)
ROI_HALF_WIDTH = 1     # 3px total width
VSPAN_TOTAL_CM = 0.7   # total vertical span (cm)
N_ABOVE = 4
N_BELOW = 4

# optional small vertical averaging height (pixels)
# 0 => 1 row, 1 => 3 rows, 2 => 5 rows ...
Y_HALF_HEIGHT_PX = 1

# Preprocess
DETREND = True
Z_SCORE = True

# For quick debug; None means full video
MAX_FRAMES = None

# Output root folder (must be this)
RESULTS_ROOT = "results_vertical"

# ============================================================
# 4) Helpers
# ============================================================
def find_video_path(base_dir: str) -> str:
    # Also accept any video file if only one exists (fallback).
    for name in VIDEO_CANDIDATES:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            return p

    # fallback: try common extensions
    exts = (".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV", ".MKV", ".AVI")
    vids = [f for f in os.listdir(base_dir) if f.endswith(exts)]
    if len(vids) == 1:
        return os.path.join(base_dir, vids[0])

    raise FileNotFoundError(
        f"No video found in {base_dir}. Put your file there (e.g., tails_pattern.mov/mp4). "
        f"Found candidates: {vids}"
    )

def load_sphere_center(center_file: str):
    """
    Expecting sphere_center_radius.npy = [xc, yc, R_pix]
    """
    if not os.path.exists(center_file):
        raise FileNotFoundError(f"Missing center file: {center_file}")
    arr = np.load(center_file)
    if arr.shape[0] < 3:
        raise ValueError(f"Center file must contain [xc, yc, R_pix]. Got shape {arr.shape}")
    xc, yc, r_pix = float(arr[0]), float(arr[1]), float(arr[2])
    return xc, yc, r_pix

def safe_slice(img, x0, x1, y0, y1):
    h, w = img.shape[:2]
    x0c = max(0, min(w, x0))
    x1c = max(0, min(w, x1))
    y0c = max(0, min(h, y0))
    y1c = max(0, min(h, y1))
    if x1c <= x0c or y1c <= y0c:
        return None
    return img[y0c:y1c, x0c:x1c]

def build_vertical_offsets_cm(vspan_total_cm: float, n_above: int, n_below: int):
    """
    Total span = vspan_total_cm, centered around yc.
    We place n_above points above (positive cm) and n_below below (negative cm),
    evenly spaced inside half-span.

    half = 0.35 cm when vspan_total_cm=0.7
    for 4 samples => [0.0875, 0.175, 0.2625, 0.35]
    """
    half = 0.5 * float(vspan_total_cm)

    def side_list(n):
        if n <= 0:
            return []
        step = half / n
        return [step * (k + 1) for k in range(n)]  # 1..n

    above = side_list(n_above)  # +cm
    below = side_list(n_below)  # -cm

    offsets = []
    # order top->bottom: above large->small, then below small->large
    for v in reversed(above):
        offsets.append(+v)
    for v in below:
        offsets.append(-v)
    return offsets

def detrend_1d(x: np.ndarray) -> np.ndarray:
    t = np.arange(len(x), dtype=np.float64)
    A = np.vstack([t, np.ones_like(t)]).T
    m, b = np.linalg.lstsq(A, x, rcond=None)[0]
    return x - (m * t + b)

def zscore_1d(x: np.ndarray, eps=1e-12) -> np.ndarray:
    mu = np.mean(x)
    sd = np.std(x)
    if sd < eps:
        return x * 0.0
    return (x - mu) / sd

def corrcoef_safe(a: np.ndarray, b: np.ndarray, eps=1e-12) -> float:
    if a.size != b.size or a.size == 0:
        return np.nan
    sa = np.std(a)
    sb = np.std(b)
    if sa < eps or sb < eps:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

# ============================================================
# 5) Single-pass extraction @ fixed r
# ============================================================
def extract_vbridge_traces_fixed_r(video_path, xc, yc, r_pix, r_cm, side="left"):
    """
    Read the video ONCE and extract 8 traces simultaneously at fixed r_cm.
    returns:
      traces: (8, T) after optional preprocess
      meta: dict
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    pix_per_cm = r_pix / SPHERE_RADIUS_CM

    # x_r computed from surface + offset (IMPORTANT)
    if side.lower() == "left":
        x_surface = xc - r_pix
        x_r = int(round(x_surface - r_cm * pix_per_cm))
    else:
        x_surface = xc + r_pix
        x_r = int(round(x_surface + r_cm * pix_per_cm))

    v_offsets_cm = build_vertical_offsets_cm(VSPAN_TOTAL_CM, N_ABOVE, N_BELOW)
    v_offsets_px = [int(round(v_cm * pix_per_cm)) for v_cm in v_offsets_cm]
    y_positions = [int(round(yc - dy)) for dy in v_offsets_px]

    N = len(y_positions)
    traces = [[] for _ in range(N)]

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if MAX_FRAMES is not None and frame_idx > MAX_FRAMES:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = gray.astype(np.float32)

        # Extract all 8 patches in this frame
        for i, y0 in enumerate(y_positions):
            x0 = x_r - ROI_HALF_WIDTH
            x1 = x_r + ROI_HALF_WIDTH + 1
            yy0 = y0 - Y_HALF_HEIGHT_PX
            yy1 = y0 + Y_HALF_HEIGHT_PX + 1
            patch = safe_slice(gray, x0, x1, yy0, yy1)
            val = float(np.mean(patch)) if patch is not None else np.nan
            traces[i].append(val)

    cap.release()
    traces = np.array(traces, dtype=np.float64)  # (8, T)

    # NaN fill
    if np.isnan(traces).any():
        for i in range(traces.shape[0]):
            x = traces[i]
            if np.isnan(x).all():
                traces[i] = 0.0
                continue
            for t in range(1, len(x)):
                if np.isnan(x[t]):
                    x[t] = x[t - 1]
            for t in range(len(x) - 2, -1, -1):
                if np.isnan(x[t]):
                    x[t] = x[t + 1]
            m = np.nanmean(x)
            x[np.isnan(x)] = m
            traces[i] = x

    # preprocess
    proc = []
    for i in range(traces.shape[0]):
        x = traces[i].copy()
        if DETREND:
            x = detrend_1d(x)
        if Z_SCORE:
            x = zscore_1d(x)
        proc.append(x)
    proc = np.array(proc, dtype=np.float64)

    meta = {
        "r_cm": float(r_cm),
        "x_r_px": int(x_r),
        "y_positions_px": [int(y) for y in y_positions],
        "fps": float(fps),
        "T": int(proc.shape[1]),
        "pix_per_cm": float(pix_per_cm),
        "roi_width_px": int(2 * ROI_HALF_WIDTH + 1),
        "vspan_total_cm": float(VSPAN_TOTAL_CM),
        "n_above": int(N_ABOVE),
        "n_below": int(N_BELOW),
        "y_half_height_px": int(Y_HALF_HEIGHT_PX),
    }
    return proc, meta

# ============================================================
# 6) Metrics + plots
# ============================================================
def compute_corr_matrix(traces: np.ndarray):
    n = traces.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            C[i, j] = 1.0 if i == j else corrcoef_safe(traces[i], traces[j])
    mask = ~np.eye(n, dtype=bool)
    mean_offdiag = float(np.mean(C[mask])) if n > 1 else 0.0
    return C, mean_offdiag

def plot_corr_matrix(C, title, out_png, out_pdf):
    plt.figure(figsize=(6, 5))
    plt.imshow(C, aspect="auto", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson r (zero-lag)")
    plt.title(title)
    plt.xlabel("Vertical sample index (top→bottom)")
    plt.ylabel("Vertical sample index (top→bottom)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()

def plot_traces_preview(traces, fps, out_png, out_pdf, max_lines=8):
    """
    Quick preview figure: all 8 traces overlayed.
    """
    T = traces.shape[1]
    t = np.arange(T) / float(fps)
    plt.figure(figsize=(8, 4))
    n = min(traces.shape[0], max_lines)
    for i in range(n):
        plt.plot(t, traces[i], label=f"v{i}")
    plt.title("VBRIDGE traces (preprocessed)")
    plt.xlabel("time (s)")
    plt.ylabel("intensity (z-scored / detrended)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()

# ============================================================
# 7) Main
# ============================================================
def main():
    global LOG_FH

    stamp = _dt.now().strftime("%Y%m%d_%H%M%S")

    # output directories
    out_root = os.path.join(BASE_DIR, RESULTS_ROOT)
    os.makedirs(out_root, exist_ok=True)
    OUT_DIR = os.path.join(out_root, f"run_{stamp}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # open log file
    LOG_FH = open(os.path.join(OUT_DIR, "run_log.txt"), "w", encoding="utf-8")

    log("START analyze_vertical_vbridge.py (FAST fixed-r design)")
    log(f"OUT_DIR = {OUT_DIR}")

    video_path = find_video_path(BASE_DIR)
    center_path = os.path.join(BASE_DIR, CENTER_FILE)
    xc, yc, r_pix = load_sphere_center(center_path)

    pix_per_cm = r_pix / SPHERE_RADIUS_CM

    log(f"SIDE={SIDE}")
    log(f"R_TARGET_CM={R_TARGET_CM} (fixed)")
    log(f"ROI_WIDTH_PX={2*ROI_HALF_WIDTH+1} (ROI_HALF_WIDTH={ROI_HALF_WIDTH})")
    log(f"VSPAN_TOTAL_CM={VSPAN_TOTAL_CM} (N_ABOVE={N_ABOVE}, N_BELOW={N_BELOW})")
    log(f"Y_HALF_HEIGHT_PX={Y_HALF_HEIGHT_PX}")
    log(f"DETREND={DETREND}, Z_SCORE={Z_SCORE}")
    log(f"MAX_FRAMES={MAX_FRAMES}")
    log(f"video_path = {video_path}")
    log(f"center_file = {center_path}")
    log(f"xc,yc,r_pix = {xc:.3f}, {yc:.3f}, {r_pix:.3f}")
    log(f"pix_per_cm = {pix_per_cm:.6f}")

    # --- single pass extraction ---
    log("Extracting 8 vertical traces (single pass) ...")
    traces, meta = extract_vbridge_traces_fixed_r(
        video_path=video_path,
        xc=xc, yc=yc, r_pix=r_pix,
        r_cm=R_TARGET_CM, side=SIDE
    )
    log(f"Done extraction. T={meta['T']} frames, fps={meta['fps']:.3f}")
    log(f"x_r_px={meta['x_r_px']}, y_positions_px(top->bottom)={meta['y_positions_px']}")

    # --- compute metrics ---
    C, mean_offdiag = compute_corr_matrix(traces)
    log(f"mean_offdiag_corr = {mean_offdiag:.6f}")

    # --- save arrays ---
    np.save(os.path.join(OUT_DIR, "traces_vbridge.npy"), traces)
    np.save(os.path.join(OUT_DIR, "corr_matrix_vbridge.npy"), C)

    # --- plots ---
    title = (
        f"VBRIDGE Corr Matrix @ r={R_TARGET_CM:.2f} cm | x={meta['x_r_px']} px | "
        f"N=8 | T={meta['T']} | ROI=3px | VSPAN=0.7cm (4 up + 4 down)"
    )
    plot_corr_matrix(
        C,
        title=title,
        out_png=os.path.join(OUT_DIR, "corr_matrix_vbridge.png"),
        out_pdf=os.path.join(OUT_DIR, "corr_matrix_vbridge.pdf"),
    )

    plot_traces_preview(
        traces=traces,
        fps=meta["fps"],
        out_png=os.path.join(OUT_DIR, "traces_preview.png"),
        out_pdf=os.path.join(OUT_DIR, "traces_preview.pdf"),
    )

    # --- run info ---
    with open(os.path.join(OUT_DIR, "run_info.txt"), "w", encoding="utf-8") as f:
        f.write("analyze_vertical_vbridge.py run info (FAST fixed-r)\n")
        f.write(f"timestamp: {stamp}\n")
        f.write(f"video_path: {video_path}\n")
        f.write(f"center_file: {center_path}\n")
        f.write(f"xc,yc,r_pix: {xc:.6f}, {yc:.6f}, {r_pix:.6f}\n")
        f.write(f"pix_per_cm: {pix_per_cm:.12f}\n")
        f.write("\n--- params ---\n")
        f.write(f"SIDE: {SIDE}\n")
        f.write(f"R_TARGET_CM: {R_TARGET_CM}\n")
        f.write(f"ROI_HALF_WIDTH: {ROI_HALF_WIDTH} (total width={2*ROI_HALF_WIDTH+1}px)\n")
        f.write(f"VSPAN_TOTAL_CM: {VSPAN_TOTAL_CM}\n")
        f.write(f"N_ABOVE/N_BELOW: {N_ABOVE}/{N_BELOW}\n")
        f.write(f"Y_HALF_HEIGHT_PX: {Y_HALF_HEIGHT_PX}\n")
        f.write(f"DETREND: {DETREND}\n")
        f.write(f"Z_SCORE: {Z_SCORE}\n")
        f.write(f"MAX_FRAMES: {MAX_FRAMES}\n")
        f.write("\n--- computed geometry ---\n")
        f.write(f"x_r_px: {meta['x_r_px']}\n")
        f.write(f"y_positions_px(top->bottom): {meta['y_positions_px']}\n")
        f.write("\n--- results ---\n")
        f.write(f"mean_offdiag_corr: {mean_offdiag:.12f}\n")

    log("DONE.")
    log(f"Output directory: {OUT_DIR}")

    # close log file
    if LOG_FH is not None:
        LOG_FH.close()
        LOG_FH = None

if __name__ == "__main__":
    main()
