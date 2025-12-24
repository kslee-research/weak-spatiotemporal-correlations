# analyze_horizontal_continuity.py
# PATCH-MODE REWRITE v3 (+ run-length histogram + max_run plots)
# + BRIDGE MODE (Fig.2 ↔ Xcorr): anchor neighborhood continuity

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Optional
import traceback

# ============================================================
# 0. Matplotlib global settings (PDF-friendly)
# ============================================================
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["figure.dpi"] = 150

# ============================================================
# 1. Basic configuration
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

# ---------------- PATCH MODE (OUR SPEC) -----------------
SIDE = "left"               # we ONLY use left side
R0_CM = 0.9                 # first reference offset from surface (cm)

# contiguous sampling / continuity mode
R_SPAN_CM = 3.0
RMAX_CM = R0_CM + R_SPAN_CM

# Patch geometry
PATCH_W = 1                 # half-width in x -> width=3 px
PATCH_H = 7                 # half-height in y -> height=15 px
GAP_PX = 1                  # 1px gap

# Spacing control
DELTA_R_CM_TARGET = 0.01

# Correlation settings
MAX_LAG_SECONDS = 3.0
MIN_Q = 0.05

# FLOW peak cutoff (tauF): ignore near-zero lags when defining "flow"
TAU_FLOW_MIN_SEC = 0.05

# Choose y-center
Y0_MODE = "yc"              # "yc" or "manual"
Y0_MANUAL = None

# ---------------- BRIDGE MODE (Fig.2 ↔ Xcorr) -----------------
BRIDGE_MODE = False

# (A) recommended: r_cm from surface
BRIDGE_R_CM_TARGET = 1.20

# (B) or: manual patch index (None이면 자동)
BRIDGE_PATCH_INDEX_MANUAL = 15

BRIDGE_NEIGHBOR_LINKS = 6
BRIDGE_DROP_INVALID = False
BRIDGE_SAVE_EXTRA_PLOTS = True

# Output
OUT_DIR = os.path.join(BASE_DIR, "results_horizental")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Logging helpers
# ============================================================
from datetime import datetime as _dt

def log(msg):
    print(f"[{_dt.now().strftime('%H:%M:%S')}] {msg}", flush=True)

log("START analyze_horizontal_continuity.py")
log(f"OUT_DIR = {OUT_DIR}")
log(f"SIDE={SIDE}, R0={R0_CM}cm, RMAX={RMAX_CM}cm, PATCH={2*PATCH_W+1}x{2*PATCH_H+1}px, GAP={GAP_PX}px")
log(f"BRIDGE_MODE={BRIDGE_MODE}, BRIDGE_R_CM_TARGET={BRIDGE_R_CM_TARGET}, BRIDGE_PATCH_INDEX_MANUAL={BRIDGE_PATCH_INDEX_MANUAL}")

def find_video_path() -> str:
    for name in VIDEO_CANDIDATES:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No video found. Tried: {VIDEO_CANDIDATES} in {BASE_DIR}")

def load_sphere_center(center_file: str):
    path = os.path.join(BASE_DIR, center_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing center file: {path}")
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.shape[0] >= 3:
        xc, yc, R_pix = float(arr[0]), float(arr[1]), float(arr[2])
        return xc, yc, R_pix
    if isinstance(arr.item(), dict):
        d = arr.item()
        return float(d["xc"]), float(d["yc"]), float(d["R_pix"])
    raise ValueError("Unexpected format in sphere_center_radius.npy")

def pix_per_cm(R_pix: float) -> float:
    return float(R_pix) / SPHERE_RADIUS_CM

def clamp_int(v, lo, hi):
    return int(max(lo, min(hi, v)))

def extract_patch_mean(frame_gray: np.ndarray, x_c: int, y_c: int, w: int, h: int) -> float:
    H, W = frame_gray.shape[:2]
    x1 = clamp_int(x_c - w, 0, W - 1)
    x2 = clamp_int(x_c + w, 0, W - 1)
    y1 = clamp_int(y_c - h, 0, H - 1)
    y2 = clamp_int(y_c + h, 0, H - 1)
    patch = frame_gray[y1:y2 + 1, x1:x2 + 1]
    if patch.size == 0:
        return float("nan")
    return float(np.mean(patch))

def normalized_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int):
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    finite = np.isfinite(a) & np.isfinite(b)
    a = a[finite]
    b = b[finite]
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    if n < 10:
        return None, None

    lags = np.arange(-max_lag, max_lag + 1, dtype=int)
    corr = np.empty_like(lags, dtype=np.float64)

    for i, lag in enumerate(lags):
        if lag >= 0:
            aa = a[: n - lag]
            bb = b[lag: n]
        else:
            aa = a[-lag: n]
            bb = b[: n + lag]
        if len(aa) < 10:
            corr[i] = np.nan
            continue
        aa = aa - np.mean(aa)
        bb = bb - np.mean(bb)
        den = (np.std(aa) * np.std(bb))
        if den <= 0:
            corr[i] = np.nan
        else:
            corr[i] = np.mean(aa * bb) / den

    return lags, corr

def pick_dual_peaks(lags: np.ndarray, corr: np.ndarray, fps: float, tau_flow_min_sec: float):
    if lags is None or corr is None:
        return (np.nan, np.nan, np.nan, np.nan)

    tau = lags.astype(np.float64) / float(fps)

    # Global maximum
    try:
        idx0 = int(np.nanargmax(corr))
        Q0 = float(corr[idx0])
        tau0 = float(tau[idx0])
    except Exception:
        Q0, tau0 = np.nan, np.nan

    # Flow maximum excluding near-zero
    maskF = np.isfinite(corr) & np.isfinite(tau) & (np.abs(tau) >= float(tau_flow_min_sec))
    if np.any(maskF):
        corrF = np.where(maskF, corr, np.nan)
        try:
            idxF = int(np.nanargmax(corrF))
            QF = float(corr[idxF])
            tauF = float(tau[idxF])
        except Exception:
            QF, tauF = np.nan, np.nan
    else:
        QF, tauF = np.nan, np.nan

    return (Q0, tau0, QF, tauF)

def build_patches_left_until_rmax(
    xc: float, yc: float, R_pix: float,
    r0_cm: float, rmax_cm: float,
    w: int, h: int, gap_px: int,
    y0: int,
    delta_r_cm_target: Optional[float] = None
):
    ppc = pix_per_cm(R_pix)
    min_dx_px = (2 * w + 1) + gap_px

    min_delta_cm = min_dx_px / ppc
    min_delta_cm = float(np.ceil(min_delta_cm * 100.0) / 100.0)

    if delta_r_cm_target is None:
        delta_r_cm = min_delta_cm
    else:
        delta_r_cm = float(delta_r_cm_target)
        if delta_r_cm < min_delta_cm:
            delta_r_cm = min_delta_cm

    x_surf = xc - R_pix
    patches = []
    k = 0
    r_cm = r0_cm

    while r_cm <= rmax_cm + 1e-12:
        xk = x_surf - (r_cm * ppc)
        xk_int = int(np.round(xk))
        yk_int = int(y0)

        bbox = (xk_int - w, xk_int + w, yk_int - h, yk_int + h)
        patches.append({"k": k, "r_cm": float(r_cm), "x_px": xk_int, "y_px": yk_int, "bbox": bbox})

        k += 1
        r_cm = r0_cm + k * delta_r_cm

    for k in range(len(patches) - 1):
        dx = abs(patches[k + 1]["x_px"] - patches[k]["x_px"])
        if dx < min_dx_px:
            raise RuntimeError(f"Non-overlap violated after rounding: dx={dx}px < {min_dx_px}px.")

    return patches, float(delta_r_cm), int(min_dx_px)

def run_length_same_sign(sign_arr: np.ndarray):
    max_run = 0
    runs = []
    cur_sign = 0
    cur_start = None

    for i, s in enumerate(sign_arr):
        if s == 0:
            if cur_sign != 0:
                runs.append((cur_start, i - 1, cur_sign))
                max_run = max(max_run, (i - cur_start))
                cur_sign = 0
                cur_start = None
            continue

        if cur_sign == 0:
            cur_sign = s
            cur_start = i
        elif s != cur_sign:
            runs.append((cur_start, i - 1, cur_sign))
            max_run = max(max_run, (i - cur_start))
            cur_sign = s
            cur_start = i

    if cur_sign != 0:
        runs.append((cur_start, len(sign_arr) - 1, cur_sign))
        max_run = max(max_run, (len(sign_arr) - cur_start))

    return int(max_run), runs

def runs_to_lengths(runs):
    L_all, L_in, L_out = [], [], []
    for a, b, s in runs:
        L = int(b - a + 1)
        L_all.append(L)
        if s > 0:
            L_in.append(L)
        elif s < 0:
            L_out.append(L)
    return L_all, L_in, L_out

def prob_run_ge(L_list, L):
    if len(L_list) == 0:
        return np.nan
    return float(np.mean(np.array(L_list, dtype=int) >= int(L)))

def pick_bridge_patch_index(patches, r_cm_target=None, manual_index=None):
    if manual_index is not None:
        k0 = int(manual_index)
        k0 = max(0, min(len(patches) - 1, k0))
        return k0, f"manual_index={manual_index}"

    if r_cm_target is None:
        return 0, "fallback=P0 (no target given)"

    r = np.array([p["r_cm"] for p in patches], dtype=np.float64)
    k0 = int(np.argmin(np.abs(r - float(r_cm_target))))
    return k0, f"nearest_to_r_cm_target={r_cm_target} (picked r={r[k0]:.3f}cm)"

# ============================================================
# 3. Main
# ============================================================
def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = os.path.join(OUT_DIR, f"patch_{stamp}")
    os.makedirs(out_base, exist_ok=True)

    def write_runlog(msg: str):
        print(msg, flush=True)
        with open(os.path.join(out_base, "run_log.txt"), "a", encoding="utf-8") as ff:
            ff.write(msg + "\n")

    try:
        write_runlog(f"[START] out_base={out_base}")

        video_path = find_video_path()
        xc, yc, R_pix = load_sphere_center(CENTER_FILE)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_lag = int(min(MAX_LAG_SECONDS * fps, 0.5 * total_frames)) if total_frames > 0 else int(MAX_LAG_SECONDS * fps)
        max_lag = max(1, max_lag)

        if Y0_MODE == "yc":
            y0 = int(np.round(yc))
        else:
            if Y0_MANUAL is None:
                raise ValueError("Y0_MODE is manual but Y0_MANUAL is None")
            y0 = int(Y0_MANUAL)

        patches, delta_r_cm, min_dx_px = build_patches_left_until_rmax(
            xc=xc, yc=yc, R_pix=R_pix,
            r0_cm=R0_CM, rmax_cm=RMAX_CM,
            w=PATCH_W, h=PATCH_H, gap_px=GAP_PX, y0=y0,
            delta_r_cm_target=DELTA_R_CM_TARGET
        )
        NUM_PATCHES = len(patches)

        write_runlog(f"Video: {video_path}")
        write_runlog(f"FPS={fps:.3f}, Frames={total_frames}")
        write_runlog(f"Center: xc={xc:.2f}, yc={yc:.2f}, R_pix={R_pix:.2f}")
        write_runlog(f"num_patches={NUM_PATCHES}, delta_r_cm_used={delta_r_cm:.2f}, min_dx_px={min_dx_px}px")

        # collect time series
        if total_frames <= 0:
            series_list = [[] for _ in range(NUM_PATCHES)]
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for k, p in enumerate(patches):
                    series_list[k].append(extract_patch_mean(gray, p["x_px"], p["y_px"], PATCH_W, PATCH_H))
                frame_idx += 1
            cap.release()
            total_frames = frame_idx
            series = [np.array(v, dtype=np.float32) for v in series_list]
        else:
            series = [np.empty(total_frames, dtype=np.float32) for _ in range(NUM_PATCHES)]
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for k, p in enumerate(patches):
                    series[k][frame_idx] = extract_patch_mean(
                        gray, x_c=p["x_px"], y_c=p["y_px"], w=PATCH_W, h=PATCH_H
                    )
                frame_idx += 1
                if frame_idx >= total_frames:
                    break
            cap.release()

            if frame_idx < total_frames:
                total_frames = frame_idx
                series = [s[:total_frames] for s in series]

        if total_frames < 10:
            raise RuntimeError(f"Too few frames read: {total_frames}")

        max_lag = int(min(MAX_LAG_SECONDS * fps, 0.5 * total_frames))
        max_lag = max(1, max_lag)

        # ========================================================
        # A) P0 vs Pk (dual-peak)
        # ========================================================
        I0 = series[0].astype(np.float64)
        results_P0 = []
        for k in range(1, NUM_PATCHES):
            Ik = series[k].astype(np.float64)
            lags, corr = normalized_xcorr(I0, Ik, max_lag=max_lag)
            if lags is None:
                results_P0.append({"k": k, "Q0": np.nan, "tau0": np.nan, "QF": np.nan, "tauF": np.nan, "v_cm_s": np.nan})
                continue

            Q0, tau0, QF, tauF = pick_dual_peaks(lags, corr, fps=fps, tau_flow_min_sec=TAU_FLOW_MIN_SEC)

            if np.isfinite(QF) and (QF >= MIN_Q) and np.isfinite(tauF) and (abs(tauF) >= TAU_FLOW_MIN_SEC):
                dr_cm = patches[k]["r_cm"] - patches[0]["r_cm"]
                v_cm_s = dr_cm / tauF
            else:
                v_cm_s = np.nan

            results_P0.append({
                "k": k,
                "Q0": float(Q0), "tau0": float(tau0),
                "QF": float(QF), "tauF": float(tauF),
                "v_cm_s": float(v_cm_s) if np.isfinite(v_cm_s) else np.nan,
            })

        # ========================================================
        # B) Adjacent links: (Pk vs Pk+1) -> continuity
        # ========================================================
        link = []
        for k in range(NUM_PATCHES - 1):
            Ia = series[k].astype(np.float64)
            Ib = series[k + 1].astype(np.float64)
            lags, corr = normalized_xcorr(Ia, Ib, max_lag=max_lag)
            if lags is None:
                link.append({"k": k, "kmid_r": np.nan, "QF": np.nan, "tauF": np.nan, "sign": 0})
                continue

            _, _, QF, tauF = pick_dual_peaks(lags, corr, fps=fps, tau_flow_min_sec=TAU_FLOW_MIN_SEC)

            valid = np.isfinite(QF) and np.isfinite(tauF) and (QF >= MIN_Q) and (abs(tauF) >= TAU_FLOW_MIN_SEC)
            sign = int(np.sign(tauF)) if valid else 0

            r_mid = 0.5 * (patches[k]["r_cm"] + patches[k + 1]["r_cm"])
            link.append({"k": k, "kmid_r": float(r_mid), "QF": float(QF), "tauF": float(tauF), "sign": sign})

        r_link = np.array([d["kmid_r"] for d in link], dtype=np.float64)
        QF_link = np.array([d["QF"] for d in link], dtype=np.float64)
        tauF_link = np.array([d["tauF"] for d in link], dtype=np.float64)
        sign_link = np.array([d["sign"] for d in link], dtype=int)

        # ========================================================
        # EXTRA SAVES: numeric packs (ADD-ONLY; does not affect PDFs)
        # ========================================================
        np.save(os.path.join(out_base, "r_link.npy"), r_link)
        np.save(os.path.join(out_base, "QF_link.npy"), QF_link)
        np.save(os.path.join(out_base, "tauF_link.npy"), tauF_link)
        np.save(os.path.join(out_base, "sign_link.npy"), sign_link)

        np.save(os.path.join(out_base, "patch_r_cm.npy"),
                np.array([p["r_cm"] for p in patches], dtype=np.float64))
        np.save(os.path.join(out_base, "patch_x_px.npy"),
                np.array([p["x_px"] for p in patches], dtype=int))
        np.save(os.path.join(out_base, "patch_y_px.npy"),
                np.array([p["y_px"] for p in patches], dtype=int))

        # OPTIONAL: store P0-vs-Pk pack for downstream aggregation
        r_vals_pack = np.array([patches[k]["r_cm"] for k in range(1, NUM_PATCHES)], dtype=np.float64)
        np.savez(
            os.path.join(out_base, "p0_vs_pk_pack.npz"),
            r_vals=r_vals_pack,
            Q0=np.array([r["Q0"] for r in results_P0], dtype=np.float64),
            tau0=np.array([r["tau0"] for r in results_P0], dtype=np.float64),
            QF=np.array([r["QF"] for r in results_P0], dtype=np.float64),
            tauF=np.array([r["tauF"] for r in results_P0], dtype=np.float64),
            v_cm_s=np.array([r["v_cm_s"] for r in results_P0], dtype=np.float64),
        )

        valid_link_mask = (sign_link != 0)
        valid_link_count = int(np.sum(valid_link_mask))
        sign_consistency = None
        if valid_link_count >= 2:
            s = sign_link[valid_link_mask]
            sign_consistency = float(np.mean(s == s[0]))

        max_run, runs = run_length_same_sign(sign_link)
        L_all, L_in, L_out = runs_to_lengths(runs)

        P_ge_3  = prob_run_ge(L_all, 3)
        P_ge_5  = prob_run_ge(L_all, 5)
        P_ge_10 = prob_run_ge(L_all, 10)
        P_in_ge_5  = prob_run_ge(L_in, 5)
        P_out_ge_5 = prob_run_ge(L_out, 5)

        max_run_all = int(max(L_all)) if len(L_all) > 0 else 0
        max_run_in  = int(max(L_in)) if len(L_in) > 0 else 0
        max_run_out = int(max(L_out)) if len(L_out) > 0 else 0

        np.savez(
            os.path.join(out_base, "runlen_pack.npz"),
            L_all=np.array(L_all, dtype=int),
            L_in=np.array(L_in, dtype=int),
            L_out=np.array(L_out, dtype=int),
            max_run=max_run,
            max_run_all=max_run_all,
            max_run_in=max_run_in,
            max_run_out=max_run_out,
            P_ge_3=P_ge_3,
            P_ge_5=P_ge_5,
            P_ge_10=P_ge_10,
            P_in_ge_5=P_in_ge_5,
            P_out_ge_5=P_out_ge_5,
            valid_link_count=valid_link_count,
            sign_consistency=(np.nan if sign_consistency is None else float(sign_consistency)),
        )

        # ========================================================
        # B-2) BRIDGE MODE
        # ========================================================
        bridge_info = None
        bridge_pack = None

        if BRIDGE_MODE:
            k0, why = pick_bridge_patch_index(
                patches,
                r_cm_target=BRIDGE_R_CM_TARGET,
                manual_index=BRIDGE_PATCH_INDEX_MANUAL
            )

            i_anchor_left = k0 - 1
            i1 = max(0, i_anchor_left - BRIDGE_NEIGHBOR_LINKS)
            i2 = min(len(link) - 1, i_anchor_left + BRIDGE_NEIGHBOR_LINKS)

            rB   = r_link[i1:i2+1].copy()
            qB   = QF_link[i1:i2+1].copy()
            tauB = tauF_link[i1:i2+1].copy()
            sB   = sign_link[i1:i2+1].copy()

            if BRIDGE_DROP_INVALID:
                m = (sB != 0)
                rB, qB, tauB, sB = rB[m], qB[m], tauB[m], sB[m]

            max_run_B, runs_B = run_length_same_sign(sB)
            L_all_B, L_in_B, L_out_B = runs_to_lengths(runs_B)

            bridge_info = {
                "k0": int(k0),
                "why": str(why),
                "r0_cm": float(patches[k0]["r_cm"]),
                "link_range": (int(i1), int(i2)),
                "max_run_B": int(max_run_B),
                "num_links_B": int(len(sB)),
                "valid_links_B": int(np.sum(sB != 0)),
                "P_ge_3_B": prob_run_ge(L_all_B, 3),
                "P_ge_5_B": prob_run_ge(L_all_B, 5),
                "P_ge_10_B": prob_run_ge(L_all_B, 10),
            }

            bridge_pack = (rB, qB, tauB, sB, runs_B)

            # extra save for bridge (only when enabled)
            np.savez(
                os.path.join(out_base, "bridge_pack.npz"),
                rB=rB, qB=qB, tauB=tauB, sB=sB,
                k0=int(bridge_info["k0"]),
                r0_cm=float(bridge_info["r0_cm"]),
                link_i1=int(bridge_info["link_range"][0]),
                link_i2=int(bridge_info["link_range"][1]),
                max_run_B=int(bridge_info["max_run_B"]),
                valid_links_B=int(bridge_info["valid_links_B"]),
                num_links_B=int(bridge_info["num_links_B"]),
                P_ge_3_B=bridge_info["P_ge_3_B"],
                P_ge_5_B=bridge_info["P_ge_5_B"],
                P_ge_10_B=bridge_info["P_ge_10_B"],
            )

        # ========================================================
        # Save outputs (summary.txt)
        # ========================================================
        with open(os.path.join(out_base, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"Video: {video_path}\n")
            f.write(f"FPS: {fps:.3f}, Frames: {total_frames}\n")
            f.write(f"Center: xc={xc:.2f}, yc={yc:.2f}, R_pix={R_pix:.2f}\n")
            f.write(f"Patch: width={2*PATCH_W+1}px, height={2*PATCH_H+1}px, gap={GAP_PX}px\n")
            f.write(f"min_dx_px={min_dx_px}px (enforced)\n")
            f.write(f"Left surface x_surf = xc - R_pix\n")
            f.write(f"r0_cm={R0_CM:.2f}, rmax_cm={RMAX_CM:.2f}, num_patches={NUM_PATCHES}, delta_r_cm_used={delta_r_cm:.2f}\n")

            f.write("\n--- Thresholds ---\n")
            f.write(f"MIN_Q={MIN_Q:.3f}\n")
            f.write(f"TAU_FLOW_MIN_SEC={TAU_FLOW_MIN_SEC:.3f}\n")

            f.write("\n--- Continuity diagnostics (adjacent links) ---\n")
            f.write(f"valid_links: {valid_link_count}/{NUM_PATCHES-1}\n")
            f.write(f"tauF_sign_consistency (among valid links): {sign_consistency}\n")
            f.write(f"max_run_length (same sign, contiguous links): {max_run}\n")
            f.write(f"runs (start_link, end_link, sign): {runs}\n")

            f.write("\n--- Run-length stats (derived from runs) ---\n")
            f.write(f"num_runs_total: {len(L_all)}\n")
            f.write(f"num_runs_inward(+): {len(L_in)}, num_runs_outward(-): {len(L_out)}\n")
            f.write(f"max_run_all: {max_run_all}\n")
            f.write(f"max_run_inward: {max_run_in}\n")
            f.write(f"max_run_outward: {max_run_out}\n")
            f.write(f"P(run>=3): {P_ge_3}\n")
            f.write(f"P(run>=5): {P_ge_5}\n")
            f.write(f"P(run>=10): {P_ge_10}\n")
            f.write(f"P(inward_run>=5): {P_in_ge_5}\n")
            f.write(f"P(outward_run>=5): {P_out_ge_5}\n")

            if BRIDGE_MODE and (bridge_info is not None):
                f.write("\n--- BRIDGE MODE (Fig.2 ↔ Xcorr) ---\n")
                f.write("BRIDGE_MODE=True\n")
                f.write(f"BRIDGE_R_CM_TARGET={BRIDGE_R_CM_TARGET}\n")
                f.write(f"BRIDGE_PATCH_INDEX_MANUAL={BRIDGE_PATCH_INDEX_MANUAL}\n")
                f.write(f"BRIDGE_NEIGHBOR_LINKS={BRIDGE_NEIGHBOR_LINKS}\n")
                f.write(f"BRIDGE_DROP_INVALID={BRIDGE_DROP_INVALID}\n")
                f.write(f"Picked anchor patch: P{bridge_info['k0']} at r={bridge_info['r0_cm']:.3f}cm\n")
                f.write(f"Reason: {bridge_info['why']}\n")
                f.write(f"Link range used: {bridge_info['link_range']}\n")
                f.write(f"Local valid_links: {bridge_info['valid_links_B']}/{bridge_info['num_links_B']}\n")
                f.write(f"Local max_run: {bridge_info['max_run_B']}\n")
                f.write(f"Local P(run>=3): {bridge_info['P_ge_3_B']}\n")
                f.write(f"Local P(run>=5): {bridge_info['P_ge_5_B']}\n")
                f.write(f"Local P(run>=10): {bridge_info['P_ge_10_B']}\n")

            f.write("\nPatch list:\n")
            for p in patches:
                f.write(f"  P{p['k']}: r={p['r_cm']:.2f}cm, x={p['x_px']}px, y={p['y_px']}px, bbox={p['bbox']}\n")

            f.write("\nResults vs P0 (dual-track):\n")
            for r in results_P0:
                f.write(
                    f"  P{r['k']}: "
                    f"Q0={r['Q0']:.4f}, tau0={r['tau0']:.6f}s, "
                    f"QF={r['QF']:.4f}, tauF={r['tauF']:.6f}s, "
                    f"v={r['v_cm_s']}\n"
                )

            f.write("\nAdjacent link results (Pk vs Pk+1):\n")
            for d in link:
                f.write(
                    f"  L{d['k']} (r_mid={d['kmid_r']:.3f}cm): "
                    f"QF={d['QF']:.4f}, tauF={d['tauF']:.6f}s, sign={d['sign']}\n"
                )

        # ========================================================
        # Plots (UNCHANGED)
        # ========================================================
        r_vals = np.array([patches[k]["r_cm"] for k in range(1, NUM_PATCHES)], dtype=np.float64)
        Q0s = np.array([r["Q0"] for r in results_P0], dtype=np.float64)
        tau0s = np.array([r["tau0"] for r in results_P0], dtype=np.float64)
        QFs = np.array([r["QF"] for r in results_P0], dtype=np.float64)
        tauFs = np.array([r["tauF"] for r in results_P0], dtype=np.float64)
        vs = np.array([r["v_cm_s"] for r in results_P0], dtype=np.float64)

        plt.figure()
        plt.plot(r_vals, Q0s, marker="o")
        plt.xlabel("r from surface (cm) [left side]")
        plt.ylabel("Q0 = peak correlation (global max)")
        plt.title("Patch-mode: Q0 vs r (compression peak)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_base, "Q0_vs_r.pdf"))
        plt.close()

        plt.figure()
        plt.plot(r_vals, tau0s, marker="o")
        plt.xlabel("r from surface (cm) [left side]")
        plt.ylabel("tau0 (s) at global peak")
        plt.title("Patch-mode: tau0 vs r (compression / synchronization)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_base, "tau0_vs_r.pdf"))
        plt.close()

        plt.figure()
        plt.plot(r_vals, QFs, marker="o")
        plt.xlabel("r from surface (cm) [left side]")
        plt.ylabel("QF = peak correlation (|tau|>=cut)")
        plt.title("Patch-mode: QF vs r (flow peak)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_base, "QF_vs_r.pdf"))
        plt.close()

        plt.figure()
        plt.plot(r_vals, tauFs, marker="o")
        plt.xlabel("r from surface (cm) [left side]")
        plt.ylabel("tauF (s) at flow peak")
        plt.title("Patch-mode: tauF vs r (flow peak)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_base, "tauF_vs_r.pdf"))
        plt.close()

        plt.figure()
        plt.plot(r_vals, vs, marker="o")
        plt.xlabel("r from surface (cm) [left side]")
        plt.ylabel("v = Δr / tauF (cm/s)")
        plt.title("Patch-mode: velocity vs r (flow)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_base, "v_vs_r.pdf"))
        plt.close()

        plt.figure()
        plt.plot(r_link, tauF_link, marker="o")
        plt.xlabel("r_mid between adjacent patches (cm)")
        plt.ylabel("tauF_link (s)  [Pk vs Pk+1]")
        plt.title("Continuity: tauF on adjacent links (sign -> inward/outward candidate)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_base, "continuity_tauF_link.pdf"))
        plt.close()

        plt.figure()
        plt.plot(r_link, QF_link, marker="o")
        plt.xlabel("r_mid between adjacent patches (cm)")
        plt.ylabel("QF_link  [Pk vs Pk+1]")
        plt.title("Continuity: QF on adjacent links")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_base, "continuity_QF_link.pdf"))
        plt.close()

        plt.figure()
        plt.plot(r_link, sign_link, marker="s")
        plt.yticks([-1, 0, 1], ["-1 (outward)", "0 (invalid)", "+1 (inward)"])
        plt.xlabel("r_mid between adjacent patches (cm)")
        plt.ylabel("sign(tauF_link)")
        plt.title(f"Continuity strip: sign along r  |  max_run={max_run}, sign_consistency={sign_consistency}",loc="left",fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_base, "continuity_sign_strip.pdf"))
        plt.close()

        if BRIDGE_MODE and (bridge_pack is not None):
            rB, qB, tauB, sB, runs_B = bridge_pack

            plt.figure()
            plt.plot(rB, sB, marker="s")
            plt.yticks([-1, 0, 1], ["-1 (outward)", "0 (invalid)", "+1 (inward)"])
            plt.xlabel("r_mid between adjacent patches (cm)")
            plt.ylabel("sign(tauF_link)")
            plt.title(
                f"BRIDGE sign strip around P{bridge_info['k0']} (r≈{bridge_info['r0_cm']:.2f}cm)\n"
                f"{bridge_info['why']} | local max_run={bridge_info['max_run_B']} | valid_links={bridge_info['valid_links_B']}/{bridge_info['num_links_B']}"
            )
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(out_base, "bridge_sign_strip.pdf"))
            plt.close()

            if BRIDGE_SAVE_EXTRA_PLOTS:
                plt.figure()
                plt.plot(rB, tauB, marker="o")
                plt.xlabel("r_mid between adjacent patches (cm)")
                plt.ylabel("tauF_link (s)")
                plt.title(
                    f"BRIDGE tauF around P{bridge_info['k0']} (r≈{bridge_info['r0_cm']:.2f}cm)\n"
                    f"local max_run={bridge_info['max_run_B']} | P>=5={bridge_info['P_ge_5_B']}"
                )
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(out_base, "bridge_tauF_link.pdf"))
                plt.close()

                plt.figure()
                plt.plot(rB, qB, marker="o")
                plt.xlabel("r_mid between adjacent patches (cm)")
                plt.ylabel("QF_link")
                plt.title(f"BRIDGE QF around P{bridge_info['k0']} (r≈{bridge_info['r0_cm']:.2f}cm)")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(out_base, "bridge_QF_link.pdf"))
                plt.close()

                L_all_B, L_in_B, L_out_B = runs_to_lengths(runs_B)
                max_possible_B = max(1, len(sB))
                binsB = np.arange(1, max_possible_B + 2) - 0.5

                plt.figure()
                if len(L_all_B) > 0:
                    plt.hist(L_all_B, bins=binsB)
                    plt.axvline(int(max(L_all_B)), linestyle="--", linewidth=2)
                plt.xlabel("run length (# adjacent links with same nonzero sign)")
                plt.ylabel("count")
                plt.title(
                    "BRIDGE run-length histogram (local)\n"
                    f"P>=5={bridge_info['P_ge_5_B']}, P>=10={bridge_info['P_ge_10_B']}"
                )
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(out_base, "bridge_runlen_hist.pdf"))
                plt.close()

        max_possible = max(1, NUM_PATCHES - 1)
        bins = np.arange(1, max_possible + 2) - 0.5

        plt.figure()
        if len(L_all) > 0:
            plt.hist(L_all, bins=bins)
            plt.axvline(max_run_all, linestyle="--", linewidth=2)
        plt.xlabel("run length (# adjacent links with same nonzero sign)")
        plt.ylabel("count")
        plt.title(
            "Run-length histogram (all signs)\n"
            f"max_run_all={max_run_all}, P>=5={P_ge_5:.3f}  P>=10={P_ge_10:.3f}"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_base, "runlen_hist_all.pdf"))
        plt.close()

        plt.figure()
        if len(L_in) > 0:
            plt.hist(L_in, bins=bins, alpha=0.8, label="inward (+)")
            plt.axvline(max_run_in, linestyle="--", linewidth=2)
        if len(L_out) > 0:
            plt.hist(L_out, bins=bins, alpha=0.8, label="outward (-)")
            plt.axvline(max_run_out, linestyle="--", linewidth=2)
        plt.xlabel("run length (# adjacent links)")
        plt.ylabel("count")
        plt.title(
            "Run-length histogram (split by sign)\n"
            f"max_in={max_run_in}, max_out={max_run_out}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_base, "runlen_hist_split.pdf"))
        plt.close()

        plt.figure()
        theta = np.linspace(0, 2*np.pi, 400)
        xs = xc + R_pix*np.cos(theta)
        ys = yc + R_pix*np.sin(theta)
        plt.plot(xs, ys)

        for p in patches:
            x1, x2, y1, y2 = p["bbox"]
            rect_x = [x1, x2, x2, x1, x1]
            rect_y = [y1, y1, y2, y2, y1]
            plt.plot(rect_x, rect_y)
            plt.text(p["x_px"], p["y_px"], f"P{p['k']}", fontsize=8)

        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.title("Patch geometry (left side)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_base, "patch_geometry.pdf"))
        plt.close()

        write_runlog(f"[OK] Saved results to: {out_base}")

    except Exception as e:
        err_path = os.path.join(out_base, "error_log.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write("Exception occurred:\n")
            f.write(str(e) + "\n\n")
            f.write(traceback.format_exc())
        print(f"[ERROR] Exception. See: {err_path}", flush=True)
        raise

if __name__ == "__main__":
    main()
