import cv2
import os
import sys

# Args
if len(sys.argv) < 3:
    print("Usage: python extract_frames.py <video_path> <output_folder> [--interval N]")
    sys.exit(1)

video_path = sys.argv[1]
output_folder = sys.argv[2]
interval = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[3] == "--interval" else 1000

if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found at path: {video_path}")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = int(frame_count / fps)

frame_interval = int(fps * (interval / 1000))  # Convert ms to frame count

frame_idx = 0
saved_idx = 0

print(f"Extracting 1 frame every {interval}ms...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_idx % frame_interval == 0:
        frame_path = os.path.join(output_folder, f"frame_{saved_idx:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        print(f"Saved {frame_path}")
        saved_idx += 1
    frame_idx += 1

cap.release()
print("✅ Frame extraction complete.")
