import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 0. Matplotlib global settings (PDF-friendly)
# ============================================================
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 150

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
PREVIEW_PDF = "sphere_center_preview_manual.pdf"

# How many boundary points to click on the sphere rim
NUM_BOUNDARY_POINTS = 8  # 5~10 정도면 충분, 원하면 나중에 바꿔도 됨


# ============================================================
# 2. Utility functions
# ============================================================
def ensure_video():
    """
    Find the first existing video among VIDEO_CANDIDATES in BASE_DIR.
    """
    for name in VIDEO_CANDIDATES:
        path = os.path.join(BASE_DIR, name)
        if os.path.isfile(path):
            print(f"[INFO] Using video: {path}")
            return path
    raise FileNotFoundError(
        f"No video found among {VIDEO_CANDIDATES} in {BASE_DIR}"
    )


def grab_frame(video_path, frame_index=None):
    """
    Grab a single frame from the video.
    If frame_index is None, use the middle frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError("Could not read frame count from video.")

    if frame_index is None:
        idx = frame_count // 2
    else:
        idx = max(0, min(frame_count - 1, int(frame_index)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame_bgr = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Failed to grab frame at index {idx}")

    print(f"[INFO] Using frame index {idx} for manual boundary picking.")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return frame_rgb, idx, frame_count


def fit_circle_least_squares(xs, ys):
    """
    Fit a circle (x^2 + y^2 + A x + B y + C = 0) in the least-squares sense.

    Given boundary points (xi, yi), we solve:
        [xi  yi  1] [A]   =  - (xi^2 + yi^2)
                       [B]
                       [C]

    Then:
        xc = -A/2,  yc = -B/2,
        R  = sqrt( (A^2 + B^2)/4 - C )

    Returns:
        xc, yc, R
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)

    if xs.size < 3:
        raise ValueError("At least 3 points are required to fit a circle.")

    A_mat = np.column_stack([xs, ys, np.ones_like(xs)])
    b_vec = -(xs**2 + ys**2)

    sol, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    A, B, C = sol

    xc = -A / 2.0
    yc = -B / 2.0
    R_sq = (A * A + B * B) / 4.0 - C
    if R_sq < 0:
        R_sq = 0.0
    R = np.sqrt(R_sq)

    return float(xc), float(yc), float(R)


def pick_circle_from_boundary(frame_rgb):
    """
    Show the frame and let the user click several boundary points on
    the sphere rim. Then fit a circle to these points.

    Returns:
        xc, yc, R_pix  (floats) or (None, None, None) on error
    """
    h, w = frame_rgb.shape[:2]

    fig, ax = plt.subplots(figsize=(6, 6 * h / w))
    ax.imshow(frame_rgb)
    ax.set_title(
        f"Click {NUM_BOUNDARY_POINTS} points on the SPHERE BOUNDARY\n"
        "(Close the window to cancel)"
    )
    ax.set_axis_off()
    plt.tight_layout()

    print(f"[INFO] Please click {NUM_BOUNDARY_POINTS} points on the sphere rim.")
    pts = plt.ginput(NUM_BOUNDARY_POINTS, timeout=-1)
    plt.close(fig)

    if len(pts) < 3:
        print("[ERROR] Less than 3 points selected. Cannot fit a circle.")
        return None, None, None

    xs, ys = zip(*pts)
    xs = np.array(xs, dtype=np.float64)
    ys = np.array(ys, dtype=np.float64)

    xc, yc, R_pix = fit_circle_least_squares(xs, ys)

    print(f"[INFO] Fitted center: xc={xc:.1f}, yc={yc:.1f}")
    print(f"[INFO] Fitted radius: R={R_pix:.1f} px")

    return xc, yc, R_pix


def save_results(frame_rgb, xc, yc, R_pix):
    """
    Save sphere_center_radius.npy and a preview PDF with the circle overlay.
    """
    h, w = frame_rgb.shape[:2]

    # Save npy
    arr = np.array([xc, yc, R_pix], dtype=np.float32)
    npy_path = os.path.join(BASE_DIR, CENTER_FILE)
    np.save(npy_path, arr)
    print(f"[SAVE] Center & radius saved to: {npy_path}")

    # Preview PDF
    fig2, ax2 = plt.subplots(figsize=(6, 6 * h / w))
    ax2.imshow(frame_rgb)
    circ = plt.Circle((xc, yc), R_pix, color="yellow", fill=False, linewidth=2)
    ax2.add_patch(circ)
    ax2.plot(xc, yc, "ro", markersize=4)
    ax2.set_title(
        f"Fitted sphere: center=({xc:.1f}, {yc:.1f}), R={R_pix:.1f} px"
    )
    ax2.set_axis_off()
    plt.tight_layout()

    pdf_path = os.path.join(BASE_DIR, PREVIEW_PDF)
    plt.savefig(pdf_path)
    plt.close(fig2)
    print(f"[SAVE] Preview PDF saved to: {pdf_path}")


# ============================================================
# 3. Main
# ============================================================
def main():
    video_path = ensure_video()

    # --- 항상 "중간 프레임" 자동 사용 ---
    frame_rgb, idx, frame_count = grab_frame(video_path, frame_index=None)
    print(f"[INFO] Video has {frame_count} frames. Using middle index {idx}.")

    # --- loop until user is satisfied with the fitted circle ---
    while True:
        xc, yc, R_pix = pick_circle_from_boundary(frame_rgb)
        if xc is None:
            print("[INFO] No valid circle fitted. Aborting.")
            return

        # quick preview
        h, w = frame_rgb.shape[:2]
        fig, ax = plt.subplots(figsize=(6, 6 * h / w))
        ax.imshow(frame_rgb)
        circ = plt.Circle((xc, yc), R_pix, color="yellow", fill=False, linewidth=2)
        ax.add_patch(circ)
        ax.plot(xc, yc, "ro", markersize=4)
        ax.set_title(
            f"Preview: center=({xc:.1f}, {yc:.1f}), R={R_pix:.1f}px\n"
            "Close this window, then answer in console: accept? (y/n)"
        )
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

        ans = input("[INPUT] Accept this fitted circle? (y/n): ").strip().lower()
        if ans == "y":
            break
        else:
            print("[INFO] Let's try again.\n")

    # --- save final results ---
    save_results(frame_rgb, xc, yc, R_pix)

    print("\n[INFO] Done.")
    print("      If the yellow circle in the PDF sits exactly on the rim,")
    print("      you can trust sphere_center_radius.npy for all analyses.")


if __name__ == "__main__":
    main()
