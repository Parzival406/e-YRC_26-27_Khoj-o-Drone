import cv2
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to input image")
args = parser.parse_args()

image = cv2.imread(args.image)

if image is None:
    raise SystemExit(f"ERROR: Could not load image: {args.image}")

print(f"Loaded image: {args.image}")
print(f"Image size: {image.shape[1]} x {image.shape[0]}")

# Step 1: Detect the four ArUco corner markers

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
aruco_params = cv2.aruco.DetectorParameters_create()

corners, ids, _ = cv2.aruco.detectMarkers(
    image,
    aruco_dict,
    parameters=aruco_params
)

required_ids = {80, 85, 90, 95}

if ids is None:
    raise SystemExit("ERROR: No ArUco markers detected.")

detected_ids = {int(marker_id) for marker_id in ids.flatten()}

print("Detected ArUco IDs:", sorted(detected_ids))

missing_ids = required_ids - detected_ids

if missing_ids:
    raise SystemExit(
        f"ERROR: Missing required ArUco markers: {sorted(missing_ids)}"
    )

print("All four required corner markers detected.")

step1_image = image.copy()
cv2.aruco.drawDetectedMarkers(step1_image, corners, ids)

print("Saved: step1_markers.jpg")

# Step 2: Straighten the arena using the four ArUco markers

# Get the centre of each required marker
marker_centres = {}

for i, marker_id in enumerate(ids.flatten()):
    marker_id = int(marker_id)

    if marker_id in required_ids:
        marker_corners = corners[i][0]
        marker_centres[marker_id] = np.mean(marker_corners, axis=0)

# Sort the four markers by their position in the image
points = list(marker_centres.items())

points.sort(key=lambda item: item[1][1])

top_two = sorted(points[:2], key=lambda item: item[1][0])
bottom_two = sorted(points[2:], key=lambda item: item[1][0])

ordered_markers = [
    top_two[0],
    top_two[1],
    bottom_two[1],
    bottom_two[0]
]

# For each marker, choose the corner facing the centre of the arena
all_marker_centres = np.array(
    [centre for _, centre in ordered_markers],
    dtype=np.float32
)

arena_centre = np.mean(all_marker_centres, axis=0)

source_points = []

for marker_id, marker_centre in ordered_markers:
    marker_index = list(ids.flatten()).index(marker_id)
    marker_corners = corners[marker_index][0]

    distances = np.linalg.norm(
        marker_corners - arena_centre,
        axis=1
    )

    inner_corner = marker_corners[np.argmin(distances)]
    source_points.append(inner_corner)

source_points = np.array(source_points, dtype=np.float32)

# Map the playing field to exactly 900 x 900 pixels
size = 900

destination_points = np.array([
    [0, 0],
    [size - 1, 0],
    [size - 1, size - 1],
    [0, size - 1]
], dtype=np.float32)

matrix = cv2.getPerspectiveTransform(
    source_points,
    destination_points
)

rectified = cv2.warpPerspective(
    image,
    matrix,
    (size, size)
)

clean_rectified = rectified.copy()


print("Arena rectified to 900 x 900.")
print("Marker order:", [marker_id for marker_id, _ in ordered_markers])
print("Saved: step2_rectified.jpg")

# Step 3: Generate the 11 x 11 grid intersections

grid_points = []

cell_size = size / 12

grid_image = clean_rectified.copy()

# Draw the 11 interior horizontal and vertical grid lines.
for i in range(1, 12):
    x = int(round(i * cell_size))
    y = int(round(i * cell_size))

    cv2.line(rectified, (x, 0), (x, size - 1), (0, 255, 0), 1)
    cv2.line(rectified, (0, y), (size - 1, y), (0, 255, 0), 1)

# Store the 121 intersections.
for row in range(1, 12):
    y = int(round(row * cell_size))

    for col in range(1, 12):
        x = int(round(col * cell_size))
        grid_points.append((x, y))


print(f"Generated {len(grid_points)} grid intersections.")
print("Saved: step3_grid.jpg")
# Step 4: Name every grid intersection

grid_names = []

for row in range(11):
    for col in range(11):
        name = chr(ord('A') + col) + str(row + 1)
        grid_names.append(name)

print(f"Generated {len(grid_names)} intersection names.")
print(f"Top-left: {grid_names[0]}")
print(f"Bottom-right: {grid_names[-1]}")
# Step 5: Detect yellow and red/orange survivors

hsv = cv2.cvtColor(clean_rectified, cv2.COLOR_BGR2HSV)

# Yellow mask
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([40, 255, 255])
yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

# Red/orange mask
red_lower = np.array([0, 80, 80])
red_upper = np.array([10, 255, 255])
red_mask = cv2.inRange(hsv, red_lower, red_upper)

# Clean small noise
kernel = np.ones((5, 5), np.uint8)

yellow_mask = cv2.morphologyEx(
    yellow_mask, cv2.MORPH_OPEN, kernel
)


yellow_contours, _ = cv2.findContours(
    yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

red_contours, _ = cv2.findContours(
    red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

# Ignore tiny regions/noise
min_area = 500

yellow_contours = [
    c for c in yellow_contours
    if cv2.contourArea(c) >= min_area
]

red_contours = [
    c for c in red_contours
    if cv2.contourArea(c) >= min_area
]

print(f"Yellow survivor outlines: {len(yellow_contours)}")
print(f"Red/orange survivor outlines: {len(red_contours)}")
print(f"Total survivor outlines: {len(yellow_contours) + len(red_contours)}")

checkpoint = grid_image.copy()

cv2.drawContours(checkpoint, yellow_contours, -1, (255, 0, 255), 3)
cv2.drawContours(checkpoint, red_contours, -1, (0, 255, 0), 3)


print("Saved: step5_outlines.jpg")


# Step 6: Find one centre point for each survivor
all_contours = yellow_contours + red_contours

centres = []

for contour in all_contours:
    M = cv2.moments(contour)

    # Handle zero-area contours safely
    if M["m00"] == 0:
        continue

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    centres.append((cx, cy))

print(f"Survivor centres: {len(centres)}")
print(centres)

step6_image = checkpoint.copy()

for cx, cy in centres:
    cv2.circle(step6_image, (cx, cy), 6, (255, 255, 255), -1)

print("Saved: step6_centres.jpg")

# Step 7: Assign each survivor to its nearest grid intersection

assigned_ids = []

for cx, cy in centres:
    distances = []

    for gx, gy in grid_points:
        distance = (cx - gx) ** 2 + (cy - gy) ** 2
        distances.append(distance)

    nearest_index = int(np.argmin(distances))
    assigned_ids.append(grid_names[nearest_index])

print("Detected marker IDs:", assigned_ids)

step7_image = step6_image.copy()

for (cx, cy), marker_id in zip(centres, assigned_ids):
    cv2.putText(
        step7_image,
        marker_id,
        (cx + 8, cy - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

print("Saved: step7_assigned.jpg")

# Write the required results file

yellow_count = len(yellow_contours)

stable_survivors = assigned_ids[:yellow_count]
critical_survivors = assigned_ids[yellow_count:]

image_path = os.path.abspath(args.image)
image_dir = os.path.dirname(image_path)
image_stem = os.path.splitext(os.path.basename(image_path))[0]

results_path = os.path.join(
    image_dir,
    image_stem + "_results.txt"
)

with open(results_path, "w") as f:
    f.write(f"Detected marker IDs: {sorted(detected_ids)}\n")
    f.write(f"Critical Survivors: {', '.join(critical_survivors)}\n")
    f.write(f"Stable Survivors: {', '.join(stable_survivors)}\n")

print(f"Saved results file: {results_path}")