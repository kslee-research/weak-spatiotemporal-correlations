# analyze_local_patch_drift.py
# Local texture patch drift analysis
# Input:
#   tails_pattern.mov
#   roi_mask_info.npy
#
# Run:
#   python analyze_local_patch_drift.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import csv
import json


VIDEO_FILE = "tails_pattern.mov"
ROI_FILE = "roi_mask_info.npy"

MAX_FRAMES = 900
DOWNSCALE = 0.35

PATCH_SIZE = 20         # block/texture patch size in downscaled pixels
PATCH_STEP = 10         # patch grid spacing (denser sampling for radial-filtered analysis)
SEARCH_RADIUS = 8       # search range in pixels
FRAME_GAP = 15           # 60 frames = about 1 sec at 60 fps
FRAME_STEP = 15          # repeat every 1 sec
MIN_TEXTURE_STD = 0.7    # reject flat patches
MIN_CORR = 0.55         # reject weak matches

# Radial-direction filter.
# radiality = |radial_component| / speed_px
# 1.0 = nearly pure inward/outward radial motion; 0.0 = tangential motion.
RADIALITY_MIN = 0.75
MIN_SPEED_FOR_DIRECTION = 0.5

# Sphere-centered visualization only. This does not change the analysis.
SPHERE_MARKER_RADIUS_PX = 28   # visual reference radius in downscaled ROI pixels
RADIAL_GUIDE_STEP_PX = 40      # spacing of guide circles
VECTOR_SCALE = 1               # quiver scale, keep 1 for true dx/dy size

# Spatial-sector analysis: 2 columns (left/right) x 3 rows (upper/middle/lower).
# This tests whether inward drift is distributed across the frame rather than
# being confined to one local patch or optical artifact region.
SECTOR_ROWS = 3
SECTOR_COLS = 2
MIN_VECTORS_PER_SECTOR = 5


def load_video_gray(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(
            gray, None,
            fx=DOWNSCALE, fy=DOWNSCALE,
            interpolation=cv2.INTER_AREA
        )
        frames.append(gray.astype(np.float32))

    cap.release()

    if len(frames) < FRAME_GAP + 10:
        raise RuntimeError("Video too short.")

    return np.stack(frames, axis=0)


def load_roi_info(path):
    return np.load(path, allow_pickle=True).item()


def extract_roi_and_mask(stack, roi_info):
    """Extract the analysis ROI and build a valid-mask from roi_mask_info.npy.

    This version supports the newer ROI file created by
    make_roi_center_opticalflow.py:
      - exclude_roi: manually selected sphere + connector / shielding rectangle
      - center: manually clicked sphere center
      - edge_exclude_fraction: outer frame boundary exclusion fraction

    Returned center_xy is expressed in the cropped/downscaled ROI coordinate
    system and is used as the true sphere center for all radial calculations.
    """
    T, H, W = stack.shape

    # -------------------------------------------------------------
    # New format: make_roi_center_opticalflow.py saves exclude_roi
    # and center in ORIGINAL video coordinates. Since stack has
    # already been downscaled, convert them by DOWNSCALE here.
    # -------------------------------------------------------------
    if "exclude_roi" in roi_info:
        full_mask = np.ones((H, W), dtype=bool)

        # 1) Exclude outer frame edges.
        edge_frac = float(roi_info.get("edge_exclude_fraction", 0.10))
        edge_x = int(W * edge_frac)
        edge_y = int(H * edge_frac)

        if edge_x > 0:
            full_mask[:, :edge_x] = False
            full_mask[:, W - edge_x:] = False
        if edge_y > 0:
            full_mask[:edge_y, :] = False
            full_mask[H - edge_y:, :] = False

        # 2) Exclude the manually selected sphere/shield/connector rectangle.
        ex = roi_info["exclude_roi"]
        ex_x1 = int(ex["x"] * DOWNSCALE)
        ex_y1 = int(ex["y"] * DOWNSCALE)
        ex_x2 = int((ex["x"] + ex["w"]) * DOWNSCALE)
        ex_y2 = int((ex["y"] + ex["h"]) * DOWNSCALE)

        ex_x1 = max(0, min(W, ex_x1))
        ex_x2 = max(0, min(W, ex_x2))
        ex_y1 = max(0, min(H, ex_y1))
        ex_y2 = max(0, min(H, ex_y2))
        full_mask[ex_y1:ex_y2, ex_x1:ex_x2] = False

        # 3) Crop to the valid inner frame boundary for compact analysis.
        x1 = edge_x
        y1 = edge_y
        x2 = W - edge_x
        y2 = H - edge_y

        x1 = max(0, min(W - 1, x1))
        x2 = max(x1 + 1, min(W, x2))
        y1 = max(0, min(H - 1, y1))
        y2 = max(y1 + 1, min(H, y2))

        roi_stack = stack[:, y1:y2, x1:x2]
        roi_mask = full_mask[y1:y2, x1:x2]

        # 4) True sphere center in cropped/downscaled ROI coordinates.
        if "center" in roi_info:
            cx_full = float(roi_info["center"]["x"]) * DOWNSCALE
            cy_full = float(roi_info["center"]["y"]) * DOWNSCALE
            center_xy = (cx_full - x1, cy_full - y1)
        else:
            center_xy = (roi_stack.shape[2] / 2, roi_stack.shape[1] / 2)

        return roi_stack, roi_mask, (x1, y1, x2, y2), center_xy

    # -------------------------------------------------------------
    # Backward-compatible path for older ROI files that directly
    # contain mask / roi_mask / valid_mask fields.
    # -------------------------------------------------------------
    mask = None
    for key in ["mask", "roi_mask", "valid_mask"]:
        if key in roi_info:
            mask = roi_info[key]
            break

    x1 = roi_info.get("x1", roi_info.get("roi_x1", 0))
    y1 = roi_info.get("y1", roi_info.get("roi_y1", 0))
    x2 = roi_info.get("x2", roi_info.get("roi_x2", W))
    y2 = roi_info.get("y2", roi_info.get("roi_y2", H))

    if "roi" in roi_info:
        roi = roi_info["roi"]
        if len(roi) == 4:
            x1, y1, x2, y2 = roi

    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    x1 = int(x1 * DOWNSCALE)
    x2 = int(x2 * DOWNSCALE)
    y1 = int(y1 * DOWNSCALE)
    y2 = int(y2 * DOWNSCALE)

    x1 = max(0, min(W - 1, x1))
    x2 = max(x1 + 1, min(W, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(y1 + 1, min(H, y2))

    roi_stack = stack[:, y1:y2, x1:x2]

    if mask is None:
        roi_mask = np.ones(roi_stack.shape[1:], dtype=bool)
    else:
        mask = mask.astype(bool)
        mask_resized = cv2.resize(
            mask.astype(np.uint8),
            (W, H),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        roi_mask = mask_resized[y1:y2, x1:x2]

    if "center" in roi_info and isinstance(roi_info["center"], dict):
        cx_full = float(roi_info["center"]["x"]) * DOWNSCALE
        cy_full = float(roi_info["center"]["y"]) * DOWNSCALE
        center_xy = (cx_full - x1, cy_full - y1)
    else:
        center_xy = (roi_stack.shape[2] / 2, roi_stack.shape[1] / 2)

    return roi_stack, roi_mask, (x1, y1, x2, y2), center_xy

def norm_corr(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    a = a - np.mean(a)
    b = b - np.mean(b)

    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if denom < 1e-8:
        return np.nan

    return float(np.sum(a * b) / denom)


def find_best_match(frame1, frame2, x, y, patch_size, search_radius, valid_mask):
    H, W = frame1.shape
    hs = patch_size // 2

    x0, x1 = x - hs, x + hs
    y0, y1 = y - hs, y + hs

    if x0 < 0 or y0 < 0 or x1 >= W or y1 >= H:
        return None

    patch = frame1[y0:y1, x0:x1]
    patch_mask = valid_mask[y0:y1, x0:x1]

    if patch.shape != (patch_size, patch_size):
        return None

    if np.mean(patch_mask) < 0.95:
        return None

    if np.std(patch) < MIN_TEXTURE_STD:
        return None

    best_corr = -999
    best_dx, best_dy = 0, 0

    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            xx = x + dx
            yy = y + dy

            sx0, sx1 = xx - hs, xx + hs
            sy0, sy1 = yy - hs, yy + hs

            if sx0 < 0 or sy0 < 0 or sx1 >= W or sy1 >= H:
                continue

            candidate_mask = valid_mask[sy0:sy1, sx0:sx1]
            if np.mean(candidate_mask) < 0.95:
                continue

            candidate = frame2[sy0:sy1, sx0:sx1]

            c = norm_corr(patch, candidate)

            if np.isfinite(c) and c > best_corr:
                best_corr = c
                best_dx = dx
                best_dy = dy

    if best_corr < MIN_CORR:
        return None

    return best_dx, best_dy, best_corr


def get_spatial_sector(x, y, W, H, center_xy=None):
    """
    Assign a patch position to one of six frame sectors:
      left_upper, left_middle, left_lower,
      right_upper, right_middle, right_lower.

    The split is done inside the downscaled ROI coordinate system.
    The x split is the sphere/ROI center. The y split is by thirds.
    """
    cx = center_xy[0] if center_xy is not None else W / 2
    col = "left" if x < cx else "right"

    if y < H / 3:
        row = "upper"
    elif y < 2 * H / 3:
        row = "middle"
    else:
        row = "lower"

    return f"{col}_{row}"


SECTOR_ORDER = [
    "left_upper", "left_middle", "left_lower",
    "right_upper", "right_middle", "right_lower",
]


def summarize_subset(subset):
    """Summary helper for a list of drift result dictionaries.

    Important convention:
      - inward/outward counts use only radially aligned drift vectors.
      - tangential and neutral vectors are counted separately.
      - inward_fraction/outward_fraction are calculated within the
        radially aligned set: inward / (inward + outward).
    """
    if len(subset) == 0:
        return {
            "n_vectors": 0,
            "mean_corr": None,
            "mean_speed_px": None,
            "mean_radial_component": None,
            "median_radial_component": None,
            "mean_radiality": None,
            "inward_count": 0,
            "outward_count": 0,
            "radial_aligned_count": 0,
            "tangential_count": 0,
            "neutral_count": 0,
            "undefined_count": 0,
            "inward_fraction": None,
            "outward_fraction": None,
            "inward_minus_outward": None,
        }

    radial = np.array([r["radial_component"] for r in subset], dtype=float)
    corr = np.array([r["corr"] for r in subset], dtype=float)
    speed = np.array([r["speed_px"] for r in subset], dtype=float)
    radiality = np.array([r.get("radiality", np.nan) for r in subset], dtype=float)
    labels = [r.get("direction_label", "undefined") for r in subset]

    inward = int(sum(lbl == "inward" for lbl in labels))
    outward = int(sum(lbl == "outward" for lbl in labels))
    tangential = int(sum(lbl == "tangential" for lbl in labels))
    neutral = int(sum(lbl == "neutral" for lbl in labels))
    undefined = int(sum(lbl == "undefined" for lbl in labels))
    radial_aligned = inward + outward

    return {
        "n_vectors": int(len(subset)),
        "mean_corr": float(np.nanmean(corr)) if len(corr) else None,
        "mean_speed_px": float(np.nanmean(speed)) if len(speed) else None,
        "mean_radial_component": float(np.nanmean(radial)) if np.any(np.isfinite(radial)) else None,
        "median_radial_component": float(np.nanmedian(radial)) if np.any(np.isfinite(radial)) else None,
        "mean_radiality": float(np.nanmean(radiality)) if np.any(np.isfinite(radiality)) else None,
        "inward_count": inward,
        "outward_count": outward,
        "radial_aligned_count": radial_aligned,
        "tangential_count": tangential,
        "neutral_count": neutral,
        "undefined_count": undefined,
        "inward_fraction": float(inward / radial_aligned) if radial_aligned > 0 else None,
        "outward_fraction": float(outward / radial_aligned) if radial_aligned > 0 else None,
        "inward_minus_outward": float((inward - outward) / radial_aligned) if radial_aligned > 0 else None,
    }

def summarize_by_sector(results):
    """Return six-sector summaries and an overall spatial-consistency score."""
    sector_summary = {}
    for sector in SECTOR_ORDER:
        subset = [r for r in results if r.get("sector") == sector]
        sector_summary[sector] = summarize_subset(subset)

    valid_sectors = [
        s for s in SECTOR_ORDER
        if sector_summary[s]["radial_aligned_count"] >= MIN_VECTORS_PER_SECTOR
        and sector_summary[s]["inward_fraction"] is not None
    ]
    inward_dominant = [
        s for s in valid_sectors
        if sector_summary[s]["inward_fraction"] > sector_summary[s]["outward_fraction"]
    ]

    consistency = {
        "sector_definition": "2 columns x 3 rows inside downscaled ROI; left/right split at ROI center; upper/middle/lower split by y thirds",
        "min_vectors_per_sector": MIN_VECTORS_PER_SECTOR,
        "valid_sector_count": len(valid_sectors),
        "inward_dominant_sector_count": len(inward_dominant),
        "inward_dominant_sector_fraction": float(len(inward_dominant) / len(valid_sectors)) if len(valid_sectors) > 0 else None,
        "inward_dominant_sectors": inward_dominant,
    }

    return sector_summary, consistency


def save_sector_summary_csv(sector_summary, outpath):
    keys = [
        "sector", "n_vectors", "mean_corr", "mean_speed_px",
        "mean_radial_component", "median_radial_component", "mean_radiality",
        "inward_count", "outward_count", "radial_aligned_count",
        "tangential_count", "neutral_count", "undefined_count",
        "inward_fraction", "outward_fraction", "inward_minus_outward",
    ]
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for sector in SECTOR_ORDER:
            row = {"sector": sector}
            row.update(sector_summary[sector])
            writer.writerow(row)


def save_sector_bar_plot(sector_summary, outpath, title):
    labels = SECTOR_ORDER
    inward_vals = [sector_summary[s]["inward_fraction"] or 0 for s in labels]
    outward_vals = [sector_summary[s]["outward_fraction"] or 0 for s in labels]

    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(10, 4.8))
    plt.bar(x - width / 2, inward_vals, width, label="inward")
    plt.bar(x + width / 2, outward_vals, width, label="outward")
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("fraction")
    plt.ylim(0, 1)
    plt.title(title + "\nSix-sector inward/outward fractions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def save_sector_map_plot(results, mask, outpath, title, center_xy=None):
    """Draw inward/outward vectors with six-sector boundaries and labels."""
    H, W = mask.shape
    cx = center_xy[0] if center_xy is not None else W / 2
    cy = center_xy[1] if center_xy is not None else H / 2

    fig, ax = plt.subplots(figsize=(9, 8))

    valid_y, valid_x = np.where(mask)
    if len(valid_x) > 0:
        ax.scatter(valid_x - cx, valid_y - cy, s=0.12, c="lightgray", alpha=0.18, linewidths=0)

    # sector boundaries: center vertical line and row-third horizontal lines
    ax.axvline(0, color="black", alpha=0.35, linewidth=1.0)
    for yy in [H / 3, 2 * H / 3]:
        ax.axhline(yy - cy, color="black", alpha=0.35, linewidth=1.0)

    # sector labels
    label_positions = {
        "left_upper": (-W * 0.25, H / 6 - cy),
        "left_middle": (-W * 0.25, H / 2 - cy),
        "left_lower": (-W * 0.25, 5 * H / 6 - cy),
        "right_upper": (W * 0.25, H / 6 - cy),
        "right_middle": (W * 0.25, H / 2 - cy),
        "right_lower": (W * 0.25, 5 * H / 6 - cy),
    }
    for sector, (lx, ly) in label_positions.items():
        ax.text(lx, ly, sector, ha="center", va="center", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"))

    sphere = plt.Circle((0, 0), SPHERE_MARKER_RADIUS_PX, fill=False, color="black", linewidth=1.8)
    ax.add_patch(sphere)
    ax.scatter([0], [0], marker="+", s=180, c="black", linewidths=2.2, zorder=5)

    inward_x, inward_y, inward_dx, inward_dy = [], [], [], []
    outward_x, outward_y, outward_dx, outward_dy = [], [], [], []

    for r in results:
        radial = r.get("radial_component", np.nan)
        if not np.isfinite(radial):
            continue
        x0 = r.get("rx_from_center", r["x"] - cx)
        y0 = r.get("ry_from_center", r["y"] - cy)
        label = r.get("direction_label", "undefined")
        if label == "inward":
            inward_x.append(x0); inward_y.append(y0); inward_dx.append(r["dx"]); inward_dy.append(r["dy"])
        elif label == "outward":
            outward_x.append(x0); outward_y.append(y0); outward_dx.append(r["dx"]); outward_dy.append(r["dy"])

    if len(inward_x) > 0:
        ax.quiver(inward_x, inward_y, inward_dx, inward_dy,
                  angles="xy", scale_units="xy", scale=VECTOR_SCALE,
                  color="blue", width=0.0045, alpha=0.82, label="inward")
    if len(outward_x) > 0:
        ax.quiver(outward_x, outward_y, outward_dx, outward_dy,
                  angles="xy", scale_units="xy", scale=VECTOR_SCALE,
                  color="red", width=0.0045, alpha=0.82, label="outward")

    ax.set_title(title + "\nSix-sector spatial distribution")
    ax.set_xlabel("x from sphere center [downscaled ROI px]")
    ax.set_ylabel("y from sphere center [downscaled ROI px]")
    ax.set_aspect("equal", adjustable="box")
    # Display orientation correction: keep the manuscript figure orientation without y-axis inversion.
    # ax.invert_yaxis()
    margin = 20
    ax.set_xlim(-cx - margin, cx + margin)
    ax.set_ylim(-cy - margin, cy + margin)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(outpath, dpi=240)
    plt.close()


def analyze_drift(stack, mask, center_xy=None):
    T, H, W = stack.shape

    results = []

    centers = []

    for y in range(PATCH_SIZE, H - PATCH_SIZE, PATCH_STEP):
        for x in range(PATCH_SIZE, W - PATCH_SIZE, PATCH_STEP):
            local_mask = mask[
                y - PATCH_SIZE // 2:y + PATCH_SIZE // 2,
                x - PATCH_SIZE // 2:x + PATCH_SIZE // 2
            ]
            if local_mask.shape != (PATCH_SIZE, PATCH_SIZE):
                continue
            if np.mean(local_mask) > 0.95:
                centers.append((x, y))

    print("Candidate patches:", len(centers))

    for t0 in range(0, T - FRAME_GAP, FRAME_STEP):
        t1 = t0 + FRAME_GAP

        f0 = stack[t0]
        f1 = stack[t1]

        print(f"Analyzing frames {t0} -> {t1}")

        for x, y in centers:
            m = find_best_match(
                f0, f1, x, y,
                PATCH_SIZE,
                SEARCH_RADIUS,
                mask
            )

            if m is None:
                continue

            dx, dy, corr = m

            # True sphere center 기준 inward/outward 계산
            cx = center_xy[0] if center_xy is not None else W / 2
            cy = center_xy[1] if center_xy is not None else H / 2

            rx = x - cx
            ry = y - cy
            rnorm = np.sqrt(rx * rx + ry * ry)

            if rnorm < 1e-8:
                radial_component = np.nan
                theta_deg = np.nan
            else:
                # outward positive, inward negative
                radial_component = (dx * rx + dy * ry) / rnorm
                theta_deg = np.degrees(np.arctan2(ry, rx))

            speed = np.sqrt(dx * dx + dy * dy)

            if np.isfinite(radial_component) and speed > 1e-8:
                radiality = abs(radial_component) / speed
            else:
                radiality = np.nan

            # Direction labels are assigned only after a radiality filter.
            # This rejects tangential motion that can blur inward/outward statistics.
            if not np.isfinite(radial_component) or not np.isfinite(radiality):
                direction_label = "undefined"
            elif speed < MIN_SPEED_FOR_DIRECTION:
                direction_label = "neutral"
            elif radiality < RADIALITY_MIN:
                direction_label = "tangential"
            elif radial_component < 0:
                direction_label = "inward"
            elif radial_component > 0:
                direction_label = "outward"
            else:
                direction_label = "neutral"

            radial_aligned = direction_label in ["inward", "outward"]

            sector = get_spatial_sector(x, y, W, H, center_xy=center_xy)

            results.append({
                "t0": t0,
                "t1": t1,
                "x": x,
                "y": y,
                "rx_from_center": rx,
                "ry_from_center": ry,
                "r_from_center_px": rnorm,
                "theta_deg": theta_deg,
                "dx": dx,
                "dy": dy,
                "speed_px": speed,
                "radial_component": radial_component,
                "radiality": radiality,
                "radial_aligned": radial_aligned,
                "direction_label": direction_label,
                "sector": sector,
                "corr": corr,
            })

    return results


def save_vector_plot(results, mask, outpath, title):
    H, W = mask.shape

    plt.figure(figsize=(10, 6))
    plt.imshow(mask, cmap="gray", alpha=0.25)

    xs = []
    ys = []
    dxs = []
    dys = []
    cs = []

    for r in results:
        xs.append(r["x"])
        ys.append(r["y"])
        dxs.append(r["dx"])
        dys.append(r["dy"])
        cs.append(r["corr"])

    if len(xs) > 0:
        plt.quiver(xs, ys, dxs, dys, cs, angles="xy", scale_units="xy", scale=1)
        plt.colorbar(label="correlation")

    # Display orientation correction: do not invert the y-axis here.
    # This keeps the saved visualization vertically consistent with the manuscript figures.
    # plt.gca().invert_yaxis()
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()



def save_sphere_centered_vector_plot(results, mask, outpath, title, center_xy=None):
    """
    Same drift results, but drawn in a sphere/ROI-centered coordinate system.
    The analysis itself is unchanged:
      - radial_component < 0 means inward toward center
      - radial_component > 0 means outward away from center

    This plot is intended only for easier visual interpretation around the sphere.
    """
    H, W = mask.shape
    cx = center_xy[0] if center_xy is not None else W / 2
    cy = center_xy[1] if center_xy is not None else H / 2

    fig, ax = plt.subplots(figsize=(9, 8))

    # Draw valid ROI boundary in center coordinates
    valid_y, valid_x = np.where(mask)
    if len(valid_x) > 0:
        ax.scatter(
            valid_x - cx,
            valid_y - cy,
            s=0.15,
            c="lightgray",
            alpha=0.20,
            linewidths=0
        )

    # Radial guide circles
    max_r = int(np.sqrt((W / 2) ** 2 + (H / 2) ** 2))
    for r in range(RADIAL_GUIDE_STEP_PX, max_r + RADIAL_GUIDE_STEP_PX, RADIAL_GUIDE_STEP_PX):
        circle = plt.Circle((0, 0), r, fill=False, color="gray", alpha=0.18, linewidth=0.8)
        ax.add_patch(circle)

    # Sphere / center marker
    sphere = plt.Circle((0, 0), SPHERE_MARKER_RADIUS_PX, fill=False, color="black", linewidth=1.5)
    ax.add_patch(sphere)
    ax.scatter([0], [0], marker="+", s=160, c="black", linewidths=2)

    inward_x, inward_y, inward_dx, inward_dy, inward_c = [], [], [], [], []
    outward_x, outward_y, outward_dx, outward_dy, outward_c = [], [], [], [], []
    neutral_x, neutral_y, neutral_dx, neutral_dy, neutral_c = [], [], [], [], []

    for r in results:
        rx = r.get("rx_from_center", r["x"] - cx)
        ry = r.get("ry_from_center", r["y"] - cy)
        radial = r["radial_component"]

        if not np.isfinite(radial):
            continue

        label = r.get("direction_label", "undefined")

        if label == "inward":
            inward_x.append(rx)
            inward_y.append(ry)
            inward_dx.append(r["dx"])
            inward_dy.append(r["dy"])
            inward_c.append(r["corr"])
        elif label == "outward":
            outward_x.append(rx)
            outward_y.append(ry)
            outward_dx.append(r["dx"])
            outward_dy.append(r["dy"])
            outward_c.append(r["corr"])
        else:
            neutral_x.append(rx)
            neutral_y.append(ry)
            neutral_dx.append(r["dx"])
            neutral_dy.append(r["dy"])
            neutral_c.append(r["corr"])

    # Use different line styles by direction. Color intensity still represents correlation.
    if len(outward_x) > 0:
        q1 = ax.quiver(
            outward_x, outward_y, outward_dx, outward_dy, outward_c,
            angles="xy", scale_units="xy", scale=VECTOR_SCALE,
            width=0.0035, alpha=0.70
        )
        cbar = fig.colorbar(q1, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("correlation")

    if len(inward_x) > 0:
        q2 = ax.quiver(
            inward_x, inward_y, inward_dx, inward_dy, inward_c,
            angles="xy", scale_units="xy", scale=VECTOR_SCALE,
            width=0.0048, alpha=0.95
        )
        if len(outward_x) == 0:
            cbar = fig.colorbar(q2, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("correlation")

    if len(neutral_x) > 0:
        ax.quiver(
            neutral_x, neutral_y, neutral_dx, neutral_dy, neutral_c,
            angles="xy", scale_units="xy", scale=VECTOR_SCALE,
            width=0.0025, alpha=0.40
        )

    ax.set_title(title + "\ncenter = sphere/ROI center, inward = toward + marker")
    ax.set_xlabel("x from center [downscaled ROI px]")
    ax.set_ylabel("y from center [downscaled ROI px]")
    ax.axhline(0, color="black", alpha=0.15, linewidth=0.8)
    ax.axvline(0, color="black", alpha=0.15, linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")

    # Display orientation correction: keep the manuscript figure orientation without y-axis inversion.
    # The quantitative analysis is unchanged; this affects visualization only.
    # ax.invert_yaxis()

    margin = 20
    ax.set_xlim(-cx - margin, cx + margin)
    ax.set_ylim(-cy - margin, cy + margin)

    # Direction legend as text
    ax.text(
        0.02, 0.02,
        f"inward vectors: {len(inward_x)} | outward vectors: {len(outward_x)}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none")
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()



def save_inward_outward_radial_vector_plot(results, mask, outpath, title, center_xy=None):
    """
    Direction-specific radial drift visualization.
      - blue arrows: inward drift toward the sphere/ROI center
      - red arrows: outward drift away from the sphere/ROI center
      - arrow length: drift speed in pixels per FRAME_GAP
      - black circle: sphere/shield mask reference
      - black +: sphere/ROI center
    """
    H, W = mask.shape
    cx = center_xy[0] if center_xy is not None else W / 2
    cy = center_xy[1] if center_xy is not None else H / 2

    fig, ax = plt.subplots(figsize=(9, 8))

    valid_y, valid_x = np.where(mask)
    if len(valid_x) > 0:
        ax.scatter(
            valid_x - cx,
            valid_y - cy,
            s=0.12,
            c="lightgray",
            alpha=0.18,
            linewidths=0
        )

    # radial guide circles
    max_r = int(np.sqrt((W / 2) ** 2 + (H / 2) ** 2))
    for rr in range(RADIAL_GUIDE_STEP_PX, max_r + RADIAL_GUIDE_STEP_PX, RADIAL_GUIDE_STEP_PX):
        guide = plt.Circle((0, 0), rr, fill=False, color="gray", alpha=0.15, linewidth=0.8)
        ax.add_patch(guide)

    # sphere/shield reference and center marker
    sphere = plt.Circle((0, 0), SPHERE_MARKER_RADIUS_PX, fill=False, color="black", linewidth=1.8)
    ax.add_patch(sphere)
    ax.scatter([0], [0], marker="+", s=180, c="black", linewidths=2.2, zorder=5)

    inward_x, inward_y, inward_dx, inward_dy = [], [], [], []
    outward_x, outward_y, outward_dx, outward_dy = [], [], [], []

    for r in results:
        radial = r.get("radial_component", np.nan)
        if not np.isfinite(radial):
            continue

        x0 = r.get("rx_from_center", r["x"] - cx)
        y0 = r.get("ry_from_center", r["y"] - cy)
        dx = r["dx"]
        dy = r["dy"]

        label = r.get("direction_label", "undefined")
        if label == "inward":
            inward_x.append(x0)
            inward_y.append(y0)
            inward_dx.append(dx)
            inward_dy.append(dy)
        elif label == "outward":
            outward_x.append(x0)
            outward_y.append(y0)
            outward_dx.append(dx)
            outward_dy.append(dy)

    if len(inward_x) > 0:
        ax.quiver(
            inward_x, inward_y, inward_dx, inward_dy,
            angles="xy", scale_units="xy", scale=VECTOR_SCALE,
            color="blue", width=0.0045, alpha=0.85,
            label="inward"
        )

    if len(outward_x) > 0:
        ax.quiver(
            outward_x, outward_y, outward_dx, outward_dy,
            angles="xy", scale_units="xy", scale=VECTOR_SCALE,
            color="red", width=0.0045, alpha=0.85,
            label="outward"
        )

    ax.set_title(title + "\nblue = inward, red = outward, arrow length = drift speed")
    ax.set_xlabel("x from sphere center [downscaled ROI px]")
    ax.set_ylabel("y from sphere center [downscaled ROI px]")
    ax.axhline(0, color="black", alpha=0.15, linewidth=0.8)
    ax.axvline(0, color="black", alpha=0.15, linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    # Display orientation correction: keep the manuscript figure orientation without y-axis inversion.
    # The quantitative analysis is unchanged; this affects visualization only.
    # ax.invert_yaxis()

    margin = 20
    ax.set_xlim(-cx - margin, cx + margin)
    ax.set_ylim(-cy - margin, cy + margin)
    ax.legend(loc="upper right")

    ax.text(
        0.02, 0.02,
        f"inward: {len(inward_x)} | outward: {len(outward_x)}",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none")
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=240)
    plt.close()

def save_radial_hist(results, outpath):
    vals = [
        r["radial_component"] for r in results
        if r.get("radial_aligned", False) and np.isfinite(r["radial_component"])
    ]

    plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=40)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Radial component [px / frame gap]\nnegative = inward, positive = outward")
    plt.ylabel("count")
    plt.title("Radially Aligned Inward / Outward Patch Drift Distribution")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_speed_hist(results, outpath):
    vals = [r["speed_px"] for r in results]

    plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=40)
    plt.xlabel("Patch displacement [px / frame gap]")
    plt.ylabel("count")
    plt.title("Patch Drift Speed Distribution")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def save_csv(results, outpath):
    if len(results) == 0:
        return

    keys = list(results[0].keys())

    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def summarize(results):
    """Overall summary across the full analyzed ROI."""
    if len(results) == 0:
        return {}
    return summarize_subset(results)



# -----------------------------------------------------------------------------
# Sphere-centered radial organization metrics
# -----------------------------------------------------------------------------
# mean_radiality alone is not sufficient, because random/noise patches can also
# satisfy a radiality threshold with respect to an arbitrary center.  The metrics
# below quantify whether radially aligned vectors are spatially organized around
# the sphere/virtual center.

RING_BINS_PX = [SPHERE_MARKER_RADIUS_PX, 40, 80, 120, 160, 220, 10**9]
NEAR_RING_MIN_PX = SPHERE_MARKER_RADIUS_PX
NEAR_RING_MAX_PX = 120
VIRTUAL_CENTER_OFFSETS_PX = [
    (0, 0),
    (-80, 0), (80, 0),
    (0, -60), (0, 60),
    (-80, -60), (-80, 60),
    (80, -60), (80, 60),
]

# Angular/radial spatial coverage metrics.
# These are more important than mean_radiality alone: noise can look radial with
# respect to many arbitrary centers, but a sphere-centered field should occupy
# many angular sectors and radial shells around the true center.
ANGULAR_BINS = 12                 # 30-degree sectors around the center
MIN_VECTORS_PER_ANGULAR_BIN = 3   # sector is considered occupied if >= this many vectors
MIN_VECTORS_PER_RADIAL_SHELL = 3  # shell is considered occupied if >= this many vectors



def _safe_balance(a, b):
    a = float(a); b = float(b)
    m = max(a, b)
    return float(min(a, b) / m) if m > 0 else None


def _normalized_entropy(counts):
    counts = np.array(counts, dtype=float)
    total = np.sum(counts)
    if total <= 0:
        return None
    p = counts[counts > 0] / total
    h = -np.sum(p * np.log(p))
    return float(h / np.log(len(counts))) if len(counts) > 1 else None


def _coefficient_of_variation(counts):
    counts = np.array(counts, dtype=float)
    mean = np.mean(counts)
    return float(np.std(counts) / mean) if mean > 0 else None


def _angular_bin_counts(radial_results, bins=ANGULAR_BINS):
    """Count radially aligned vectors by polar angle around the current center."""
    counts = np.zeros(bins, dtype=int)
    for r in radial_results:
        theta = r.get("theta_deg", np.nan)
        if not np.isfinite(theta):
            continue
        theta360 = (theta + 360.0) % 360.0
        idx = int(np.floor(theta360 / (360.0 / bins)))
        idx = max(0, min(bins - 1, idx))
        counts[idx] += 1
    return counts


def _coverage_fraction(counts, min_count):
    counts = np.array(counts, dtype=float)
    if len(counts) == 0:
        return None
    return float(np.sum(counts >= min_count) / len(counts))


def _nonzero_fraction(counts):
    counts = np.array(counts, dtype=float)
    if len(counts) == 0:
        return None
    return float(np.sum(counts > 0) / len(counts))


def _organization_index(angular_coverage, angular_entropy, radial_shell_coverage,
                        lr_balance, ul_balance, near_ring_fraction):
    """Composite descriptive index in [0, 1] when inputs are available."""
    vals = [angular_coverage, angular_entropy, radial_shell_coverage,
            lr_balance, ul_balance, near_ring_fraction]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if len(vals) == 0:
        return None
    return float(np.prod(vals) ** (1.0 / len(vals)))


def relabel_results_for_center(results, W, H, cx, cy):
    """Recompute radial labels for an arbitrary center using already measured dx/dy.

    This does not redo patch matching. It asks whether the same displacement field
    appears radially organized with respect to the sphere center or an arbitrary
    virtual center.
    """
    out = []
    for r in results:
        rx = r["x"] - cx
        ry = r["y"] - cy
        rnorm = np.sqrt(rx * rx + ry * ry)
        dx = r["dx"]
        dy = r["dy"]
        speed = np.sqrt(dx * dx + dy * dy)

        if rnorm < 1e-8:
            radial_component = np.nan
            theta_deg = np.nan
        else:
            radial_component = (dx * rx + dy * ry) / rnorm
            theta_deg = np.degrees(np.arctan2(ry, rx))

        if np.isfinite(radial_component) and speed > 1e-8:
            radiality = abs(radial_component) / speed
        else:
            radiality = np.nan

        if not np.isfinite(radial_component) or not np.isfinite(radiality):
            direction_label = "undefined"
        elif speed < MIN_SPEED_FOR_DIRECTION:
            direction_label = "neutral"
        elif radiality < RADIALITY_MIN:
            direction_label = "tangential"
        elif radial_component < 0:
            direction_label = "inward"
        elif radial_component > 0:
            direction_label = "outward"
        else:
            direction_label = "neutral"

        rr = dict(r)
        rr["rx_from_center"] = rx
        rr["ry_from_center"] = ry
        rr["r_from_center_px"] = rnorm
        rr["theta_deg"] = theta_deg
        rr["radial_component"] = radial_component
        rr["radiality"] = radiality
        rr["direction_label"] = direction_label
        rr["radial_aligned"] = direction_label in ["inward", "outward"]
        # Sector is also recomputed around the chosen center; rows still use ROI thirds.
        rr["sector"] = get_spatial_sector(r["x"], r["y"], W, H, center_xy=(cx, cy))
        out.append(rr)
    return out


def summarize_radial_spatial_organization(radial_results, mask, center_label="sphere_center", center_xy=None):
    """Spatial organization summary for radially aligned vectors.

    High mean_radiality alone is not diagnostic. A random/noise cluster can have
    high radiality around many arbitrary centers. The useful question is whether
    the radially aligned vectors occupy the image plane in a sphere-centered way:
    angular coverage, radial-shell coverage, and spatial balance.
    """
    H, W = mask.shape

    counts = {s: 0 for s in SECTOR_ORDER}
    for r in radial_results:
        counts[r.get("sector", get_spatial_sector(r["x"], r["y"], W, H, center_xy=center_xy))] += 1

    sector_counts = [counts[s] for s in SECTOR_ORDER]
    left_total = counts["left_upper"] + counts["left_middle"] + counts["left_lower"]
    right_total = counts["right_upper"] + counts["right_middle"] + counts["right_lower"]
    upper_total = counts["left_upper"] + counts["right_upper"]
    middle_total = counts["left_middle"] + counts["right_middle"]
    lower_total = counts["left_lower"] + counts["right_lower"]

    total = int(len(radial_results))
    rs = np.array([r.get("r_from_center_px", np.nan) for r in radial_results], dtype=float)
    finite_rs = rs[np.isfinite(rs)]
    near_count = int(np.sum((finite_rs >= NEAR_RING_MIN_PX) & (finite_rs <= NEAR_RING_MAX_PX)))

    ring_counts = {}
    radial_shell_counts = []
    for a, b in zip(RING_BINS_PX[:-1], RING_BINS_PX[1:]):
        key = f"r_{int(a)}_{int(b) if b < 10**8 else 'inf'}"
        c = int(np.sum((finite_rs >= a) & (finite_rs < b)))
        ring_counts[key] = c
        radial_shell_counts.append(c)

    angular_counts = _angular_bin_counts(radial_results, ANGULAR_BINS)
    angular_counts_dict = {
        f"a_{int(i * 360 / ANGULAR_BINS)}_{int((i + 1) * 360 / ANGULAR_BINS)}": int(c)
        for i, c in enumerate(angular_counts)
    }

    max_sector = max(sector_counts) if sector_counts else 0
    min_sector = min(sector_counts) if sector_counts else 0

    angular_entropy = _normalized_entropy(angular_counts)
    angular_coverage = _coverage_fraction(angular_counts, MIN_VECTORS_PER_ANGULAR_BIN)
    radial_shell_entropy = _normalized_entropy(radial_shell_counts)
    radial_shell_coverage = _coverage_fraction(radial_shell_counts, MIN_VECTORS_PER_RADIAL_SHELL)
    lr_balance = _safe_balance(left_total, right_total)
    ul_balance = _safe_balance(upper_total, lower_total)
    near_fraction = float(near_count / total) if total > 0 else None

    org_index = _organization_index(
        angular_coverage,
        angular_entropy,
        radial_shell_coverage,
        lr_balance,
        ul_balance,
        near_fraction,
    )

    return {
        "center_label": center_label,
        "radial_vector_count": total,
        "sector_counts": counts,
        "left_total": int(left_total),
        "right_total": int(right_total),
        "left_right_balance_min_over_max": lr_balance,
        "upper_total": int(upper_total),
        "middle_total": int(middle_total),
        "lower_total": int(lower_total),
        "upper_lower_balance_min_over_max": ul_balance,
        "sector_entropy_normalized": _normalized_entropy(sector_counts),
        "sector_cv": _coefficient_of_variation(sector_counts),
        "sector_max_min_ratio": float(max_sector / min_sector) if min_sector > 0 else None,
        "empty_sector_count": int(sum(c == 0 for c in sector_counts)),
        "valid_sector_count_min_vectors": int(sum(c >= MIN_VECTORS_PER_SECTOR for c in sector_counts)),
        "near_ring_min_px": NEAR_RING_MIN_PX,
        "near_ring_max_px": NEAR_RING_MAX_PX,
        "near_ring_count": near_count,
        "near_ring_fraction": near_fraction,
        "ring_counts": ring_counts,
        "radial_shell_entropy_normalized": radial_shell_entropy,
        "radial_shell_coverage_fraction": radial_shell_coverage,
        "radial_shell_nonzero_fraction": _nonzero_fraction(radial_shell_counts),
        "angular_bins": ANGULAR_BINS,
        "min_vectors_per_angular_bin": MIN_VECTORS_PER_ANGULAR_BIN,
        "angular_counts": angular_counts_dict,
        "angular_entropy_normalized": angular_entropy,
        "angular_coverage_fraction": angular_coverage,
        "angular_nonzero_fraction": _nonzero_fraction(angular_counts),
        "angular_cv": _coefficient_of_variation(angular_counts),
        "sphere_centered_radial_organization_index": org_index,
    }


def save_spatial_organization_csv(spatial_summary, outpath):
    row = dict(spatial_summary)
    sector_counts = row.pop("sector_counts", {})
    ring_counts = row.pop("ring_counts", {})
    angular_counts = row.pop("angular_counts", {})
    row.update({f"sector_{k}": v for k, v in sector_counts.items()})
    row.update({f"ring_{k}": v for k, v in ring_counts.items()})
    row.update({f"angular_{k}": v for k, v in angular_counts.items()})
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_angular_coverage_plot(radial_results, outpath, title):
    counts = _angular_bin_counts(radial_results, ANGULAR_BINS)
    angles = np.arange(ANGULAR_BINS) * (360 / ANGULAR_BINS)
    labels = [f"{int(a)}-{int(a + 360 / ANGULAR_BINS)}" for a in angles]
    plt.figure(figsize=(9, 4.5))
    plt.bar(np.arange(ANGULAR_BINS), counts)
    plt.xticks(np.arange(ANGULAR_BINS), labels, rotation=45, ha="right")
    plt.xlabel("Angular sector around selected center [deg]")
    plt.ylabel("radially aligned vector count")
    plt.title(title + "\nAngular coverage of radially aligned vectors")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()


def save_radial_shell_coverage_plot(radial_results, outpath, title):
    rs = np.array([r.get("r_from_center_px", np.nan) for r in radial_results], dtype=float)
    finite_rs = rs[np.isfinite(rs)]
    labels = []
    counts = []
    for a, b in zip(RING_BINS_PX[:-1], RING_BINS_PX[1:]):
        labels.append(f"{int(a)}-{int(b) if b < 10**8 else 'inf'}")
        counts.append(int(np.sum((finite_rs >= a) & (finite_rs < b))))
    plt.figure(figsize=(8, 4.5))
    plt.bar(np.arange(len(counts)), counts)
    plt.xticks(np.arange(len(counts)), labels, rotation=45, ha="right")
    plt.xlabel("Radial shell from selected center [px]")
    plt.ylabel("radially aligned vector count")
    plt.title(title + "\nRadial-shell coverage of radially aligned vectors")
    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close()



def summarize_virtual_centers(results, mask, center_xy=None):
    """Compare sphere center against arbitrary virtual centers.

    If noise obtains high radiality for any arbitrary center, then mean_radiality
    is not diagnostic.  The useful signal should be a center-specific increase in
    balanced radial organization around the true sphere center.
    """
    H, W = mask.shape
    base_cx = center_xy[0] if center_xy is not None else W / 2
    base_cy = center_xy[1] if center_xy is not None else H / 2
    rows = []
    for dx0, dy0 in VIRTUAL_CENTER_OFFSETS_PX:
        cx = base_cx + dx0
        cy = base_cy + dy0
        relabeled = relabel_results_for_center(results, W, H, cx, cy)
        radial = [r for r in relabeled if r.get("radial_aligned", False)]
        base = summarize_subset(radial)
        spatial = summarize_radial_spatial_organization(radial, mask, center_label=f"offset_{dx0}_{dy0}", center_xy=(cx, cy))
        rows.append({
            "center_offset_x": dx0,
            "center_offset_y": dy0,
            "radial_aligned_count": base["radial_aligned_count"],
            "mean_radiality": base["mean_radiality"],
            "mean_speed_px": base["mean_speed_px"],
            "inward_fraction": base["inward_fraction"],
            "outward_fraction": base["outward_fraction"],
            "left_right_balance_min_over_max": spatial["left_right_balance_min_over_max"],
            "sector_entropy_normalized": spatial["sector_entropy_normalized"],
            "sector_cv": spatial["sector_cv"],
            "empty_sector_count": spatial["empty_sector_count"],
            "near_ring_fraction": spatial["near_ring_fraction"],
            "near_ring_count": spatial["near_ring_count"],
            "angular_coverage_fraction": spatial["angular_coverage_fraction"],
            "angular_entropy_normalized": spatial["angular_entropy_normalized"],
            "angular_cv": spatial["angular_cv"],
            "radial_shell_coverage_fraction": spatial["radial_shell_coverage_fraction"],
            "radial_shell_entropy_normalized": spatial["radial_shell_entropy_normalized"],
            "sphere_centered_radial_organization_index": spatial["sphere_centered_radial_organization_index"],
        })
    return rows


def save_virtual_center_control_csv(rows, outpath):
    if not rows:
        return
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    video_path = Path(VIDEO_FILE)
    roi_path = Path(ROI_FILE)

    if not video_path.exists():
        raise FileNotFoundError(f"{VIDEO_FILE} not found.")
    if not roi_path.exists():
        raise FileNotFoundError(f"{ROI_FILE} not found. Run make_roi_center_opticalflow.py first.")

    kind = input("Enter video type [noise/tungsten]: ").strip().lower()
    if kind not in ["noise", "tungsten"]:
        raise ValueError("Enter only noise or tungsten.")

    video_no = input("Enter video number [1,2,3...]: ").strip()
    if not video_no.isdigit():
        raise ValueError("Video number must be numeric.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("results_local_patch_drift") / f"{kind}_{video_no}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    print("[1] Loading video...")
    stack = load_video_gray(video_path)
    print("Video stack:", stack.shape)

    print("[2] Loading ROI...")
    roi_info = load_roi_info(roi_path)

    print("[3] Applying ROI/mask...")
    roi_stack, roi_mask, roi_box, center_xy = extract_roi_and_mask(stack, roi_info)
    print("ROI stack:", roi_stack.shape)
    print("ROI mask valid:", np.mean(roi_mask))
    print("Sphere center in ROI:", center_xy)

    print("[4] Local patch drift analysis...")
    results = analyze_drift(roi_stack, roi_mask, center_xy=center_xy)

    print("[5] Saving outputs...")
    save_csv(results, outdir / "patch_drift_table.csv")
    radial_results = [r for r in results if r.get("radial_aligned", False)]
    save_csv(radial_results, outdir / "patch_drift_radial_aligned_only.csv")

    save_vector_plot(
        results,
        roi_mask,
        outdir / "patch_drift_vectors.png",
        f"{kind} {video_no} - Local Patch Drift Vectors"
    )

    save_sphere_centered_vector_plot(
        radial_results,
        roi_mask,
        outdir / "sphere_centered_patch_drift_vectors.png",
        f"{kind} {video_no} - Sphere-Centered Patch Drift Vectors",
        center_xy=center_xy
    )

    save_inward_outward_radial_vector_plot(
        radial_results,
        roi_mask,
        outdir / "inward_outward_radial_drift.png",
        f"{kind} {video_no} - Inward/Outward Radial Patch Drift",
        center_xy=center_xy
    )

    save_radial_hist(
        radial_results,
        outdir / "inward_outward_histogram.png"
    )

    save_speed_hist(
        results,
        outdir / "patch_speed_histogram.png"
    )

    sector_summary, sector_consistency = summarize_by_sector(radial_results)
    save_sector_summary_csv(
        sector_summary,
        outdir / "sector_inward_outward_summary.csv"
    )
    save_sector_bar_plot(
        sector_summary,
        outdir / "sector_inward_outward_fraction.png",
        f"{kind} {video_no} - Spatial Sector Drift Summary"
    )
    save_sector_map_plot(
        radial_results,
        roi_mask,
        outdir / "six_sector_inward_outward_map.png",
        f"{kind} {video_no} - Six-Sector Inward/Outward Map",
        center_xy=center_xy
    )

    spatial_organization = summarize_radial_spatial_organization(
        radial_results,
        roi_mask,
        center_label="sphere_or_virtual_center",
        center_xy=center_xy
    )
    save_spatial_organization_csv(
        spatial_organization,
        outdir / "radial_spatial_organization_summary.csv"
    )
    save_angular_coverage_plot(
        radial_results,
        outdir / "angular_coverage_radial_vectors.png",
        f"{kind} {video_no} - Angular Coverage"
    )
    save_radial_shell_coverage_plot(
        radial_results,
        outdir / "radial_shell_coverage.png",
        f"{kind} {video_no} - Radial Shell Coverage"
    )

    virtual_center_control = summarize_virtual_centers(results, roi_mask, center_xy=center_xy)
    save_virtual_center_control_csv(
        virtual_center_control,
        outdir / "virtual_center_control_summary.csv"
    )

    summary = summarize(results)
    radial_summary = summarize(radial_results)

    settings = {
        "video_file": VIDEO_FILE,
        "roi_file": ROI_FILE,
        "kind": kind,
        "video_no": video_no,
        "MAX_FRAMES": MAX_FRAMES,
        "DOWNSCALE": DOWNSCALE,
        "PATCH_SIZE": PATCH_SIZE,
        "PATCH_STEP": PATCH_STEP,
        "SEARCH_RADIUS": SEARCH_RADIUS,
        "FRAME_GAP": FRAME_GAP,
        "FRAME_STEP": FRAME_STEP,
        "MIN_TEXTURE_STD": MIN_TEXTURE_STD,
        "MIN_CORR": MIN_CORR,
        "RADIALITY_MIN": RADIALITY_MIN,
        "MIN_SPEED_FOR_DIRECTION": MIN_SPEED_FOR_DIRECTION,
        "SPHERE_MARKER_RADIUS_PX": SPHERE_MARKER_RADIUS_PX,
        "RADIAL_GUIDE_STEP_PX": RADIAL_GUIDE_STEP_PX,
        "VECTOR_SCALE": VECTOR_SCALE,
        "SECTOR_ROWS": SECTOR_ROWS,
        "SECTOR_COLS": SECTOR_COLS,
        "MIN_VECTORS_PER_SECTOR": MIN_VECTORS_PER_SECTOR,
        "ANGULAR_BINS": ANGULAR_BINS,
        "MIN_VECTORS_PER_ANGULAR_BIN": MIN_VECTORS_PER_ANGULAR_BIN,
        "MIN_VECTORS_PER_RADIAL_SHELL": MIN_VECTORS_PER_RADIAL_SHELL,
        "RING_BINS_PX": RING_BINS_PX,
        "NEAR_RING_MIN_PX": NEAR_RING_MIN_PX,
        "NEAR_RING_MAX_PX": NEAR_RING_MAX_PX,
        "roi_box_after_downscale": roi_box,
        "sphere_center_xy_in_downscaled_roi": [float(center_xy[0]), float(center_xy[1])],
        "summary_all_vectors": summary,
        "summary_radial_aligned_only": radial_summary,
        "sector_summary": sector_summary,
        "sector_consistency": sector_consistency,
        "radial_spatial_organization": spatial_organization,
        "virtual_center_control": virtual_center_control,
    }

    with open(outdir / "settings.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    print("Summary, all vectors:")
    print(json.dumps(summary, indent=2))
    print("Summary, radial-aligned only:")
    print(json.dumps(radial_summary, indent=2))
    print("Sector consistency:")
    print(json.dumps(sector_consistency, indent=2))
    print("Radial spatial organization:")
    print(json.dumps(spatial_organization, indent=2))

    print("Done.")
    print("Output:", outdir)


if __name__ == "__main__":
    main()