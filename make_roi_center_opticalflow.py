import cv2
import numpy as np
import os

VIDEO_BASENAME = "tails_pattern"
OUTPUT_FILE = "roi_mask_info.npy"
PREVIEW_FILE = "roi_mask_preview.png"

MAX_DISPLAY_WIDTH = 1200
EDGE_EXCLUDE_FRACTION = 0.10

video_path = None
for ext in [".mp4", ".mov", ".avi", ".mkv"]:
    if os.path.exists(VIDEO_BASENAME + ext):
        video_path = VIDEO_BASENAME + ext
        break

if video_path is None:
    raise FileNotFoundError("tails_pattern video file not found.")

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Could not read first frame from video.")

frame_h, frame_w = frame.shape[:2]

# ==========================================
# Display scaling for high-resolution videos
# ==========================================
if frame_w > MAX_DISPLAY_WIDTH:
    display_scale = MAX_DISPLAY_WIDTH / frame_w
else:
    display_scale = 1.0

display = cv2.resize(
    frame,
    None,
    fx=display_scale,
    fy=display_scale,
    interpolation=cv2.INTER_AREA
)

display_h, display_w = display.shape[:2]

print("Video loaded:", video_path)
print(f"Original size : {frame_w} x {frame_h}")
print(f"Display size  : {display_w} x {display_h}")
print(f"Display scale : {display_scale:.3f}")

# ==========================================
# Step 1: Select excluded rectangle ROI
# ==========================================
print("\nStep 1: Select rectangle ROI that contains sphere + connector.")
print("Drag rectangle, then press ENTER or SPACE.")

roi_small = cv2.selectROI(
    "Select sphere+connector ROI",
    display,
    fromCenter=False,
    showCrosshair=True
)
cv2.destroyAllWindows()

x_small, y_small, w_small, h_small = map(int, roi_small)

if w_small <= 0 or h_small <= 0:
    raise RuntimeError("Invalid ROI selected.")

# Convert display coordinates to original coordinates
x = int(x_small / display_scale)
y = int(y_small / display_scale)
w = int(w_small / display_scale)
h = int(h_small / display_scale)

# Clip ROI to original frame boundary
x = max(0, min(x, frame_w - 1))
y = max(0, min(y, frame_h - 1))
w = max(1, min(w, frame_w - x))
h = max(1, min(h, frame_h - y))

print("\nROI conversion")
print("----------------")
print(f"Display ROI  : x={x_small}, y={y_small}, w={w_small}, h={h_small}")
print(f"Original ROI : x={x}, y={y}, w={w}, h={h}")

# ==========================================
# Step 2: Click sphere center
# ==========================================
display2 = display.copy()

cv2.rectangle(
    display2,
    (x_small, y_small),
    (x_small + w_small, y_small + h_small),
    (0, 255, 255),
    2
)

center_point = []

def click_center(event, cx, cy, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        center_point.clear()
        center_point.append((cx, cy))

        img = display2.copy()
        cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(
            img,
            "center",
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        cv2.imshow("Click sphere center", img)

print("\nStep 2: Click sphere center point, then press any key.")
cv2.imshow("Click sphere center", display2)
cv2.setMouseCallback("Click sphere center", click_center)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(center_point) == 0:
    raise RuntimeError("Sphere center was not selected.")

center_x_small, center_y_small = center_point[0]

# Convert display center to original coordinates
center_x = int(center_x_small / display_scale)
center_y = int(center_y_small / display_scale)

center_x = max(0, min(center_x, frame_w - 1))
center_y = max(0, min(center_y, frame_h - 1))

print("\nCenter conversion")
print("----------------")
print(f"Display center  : x={center_x_small}, y={center_y_small}")
print(f"Original center : x={center_x}, y={center_y}")

# ==========================================
# Save ROI / center information
# ==========================================
info = {
    "video_path": video_path,
    "frame_width": frame_w,
    "frame_height": frame_h,
    "display_scale": display_scale,
    "exclude_roi": {
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h)
    },
    "center": {
        "x": int(center_x),
        "y": int(center_y)
    },
    "edge_exclude_fraction": EDGE_EXCLUDE_FRACTION
}

np.save(OUTPUT_FILE, info)

# ==========================================
# Save preview image on original frame
# ==========================================
preview = frame.copy()

edge_x = int(frame_w * EDGE_EXCLUDE_FRACTION)
edge_y = int(frame_h * EDGE_EXCLUDE_FRACTION)

# Blue: valid inner frame boundary
cv2.rectangle(
    preview,
    (edge_x, edge_y),
    (frame_w - edge_x, frame_h - edge_y),
    (255, 0, 0),
    3
)

# Yellow: excluded ROI
cv2.rectangle(
    preview,
    (x, y),
    (x + w, y + h),
    (0, 255, 255),
    3
)

# Red: sphere center
cv2.circle(preview, (center_x, center_y), 10, (0, 0, 255), -1)

cv2.putText(
    preview,
    "blue: valid inner frame boundary",
    (40, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (255, 0, 0),
    3
)

cv2.putText(
    preview,
    "yellow: excluded sphere/connector ROI",
    (40, 105),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 255, 255),
    3
)

cv2.putText(
    preview,
    "red: sphere center",
    (40, 150),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 0, 255),
    3
)

cv2.imwrite(PREVIEW_FILE, preview)

print("\nSaved files")
print("-----------")
print("Saved:", OUTPUT_FILE)
print("Saved:", PREVIEW_FILE)

print("\nFinal settings")
print("--------------")
print("Video:", video_path)
print("Center:", center_x, center_y)
print("Excluded ROI:", x, y, w, h)
print("Edge exclusion fraction:", EDGE_EXCLUDE_FRACTION)