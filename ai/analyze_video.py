import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
import pandas as pd
import json
import mediapipe as mp
import matplotlib.pyplot as plt

# Set video path
default_video_path = "Jamie_Deadlift.mov"
video_path = os.path.join(".", default_video_path)

# Check if video exists
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

# Reload video
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"🎥 Detected FPS: {fps}")

# Read first frame to check orientation
ret, test_frame = cap.read()
if not ret:
    raise ValueError("Couldn't read video")

frame_height, frame_width = test_frame.shape[:2]
rotate_video = frame_width > frame_height

if rotate_video:
    out_width, out_height = frame_height, frame_width
else:
    out_width, out_height = frame_width, frame_height

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Output paths
output_base = os.path.splitext(os.path.basename(video_path))[0]
rep_video_path = f"outputs/{output_base}_rep_annotated_output.mp4"
csv_path = f"outputs/{output_base}_rep_data.csv"
json_path = f"outputs/{output_base}_rep_data.json"

# Smart MediaPipe Landmark Tracking
mp_pose = mp.solutions.pose
landmark_dict = {
    "left_wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
    "right_ankle": mp_pose.PoseLandmark.RIGHT_ANKLE,
    "hip": mp_pose.PoseLandmark.LEFT_HIP,
    "head": mp_pose.PoseLandmark.NOSE
}

landmark_positions = {k: [] for k in landmark_dict.keys()}
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            for name, lm in landmark_dict.items():
                y_pos = results.pose_landmarks.landmark[lm].y
                landmark_positions[name].append(y_pos)

cap.release()

# Select the landmark with the highest total movement
total_displacements = {k: np.sum(np.abs(np.diff(v))) for k, v in landmark_positions.items() if len(v) > 1}
best_landmark = max(total_displacements, key=total_displacements.get)
print(f"✅ Best tracking landmark selected: {best_landmark}")
raw_y = np.array(landmark_positions[best_landmark])

# Smooth and time conversion
smooth_y = np.convolve(raw_y, np.ones(5)/5, mode='valid')
time_axis = np.arange(len(smooth_y)) / fps

# Detect reps
rep_frames = []
state = "down"
threshold = 0.003

for i in range(1, len(smooth_y)):
    if state == "down" and smooth_y[i] > smooth_y[i - 1] + threshold:
        state = "up"
        rep_frames.append({"start": i})
    elif state == "up" and smooth_y[i] < smooth_y[i - 1] - threshold:
        state = "down"
        if rep_frames and "peak" not in rep_frames[-1]:
            peak_idx = np.argmax(smooth_y[rep_frames[-1]["start"]:i]) + rep_frames[-1]["start"]
            rep_frames[-1]["peak"] = peak_idx
            rep_frames[-1]["stop"] = i

# Adjust data
rep_data = []
for idx, rep in enumerate(rep_frames):
    if "start" in rep and "peak" in rep and "stop" in rep:
        start = rep["start"]
        peak = rep["peak"]
        stop = rep["stop"]

        # Always compute as positive duration (concentric = start -> peak)
        duration_sec = abs(peak - start) / fps

        concentric = duration_sec
        eccentric = concentric * 1.2
        total_tut = round(concentric + eccentric, 2)

        if duration_sec >= 3.50:
            rpe = 10.0
        elif duration_sec >= 3.00:
            rpe = 9.5
        elif duration_sec >= 2.50:
            rpe = 9.0
        elif duration_sec >= 2.00:
            rpe = 8.5
        elif duration_sec >= 1.50:
            rpe = 8.0
        elif duration_sec >= 1.00:
            rpe = 7.5
        else:
            rpe = 7.0

        rir_lookup = {
            10.0: "(Possibly 0 Reps in the Tank)",
            9.5: "(Possibly 0-1 Reps in the Tank)",
            9.0: "(Possibly 1-2 Reps in the Tank)",
            8.5: "(Possibly 2-3 Reps in the Tank)",
            8.0: "(Possibly 3-4 Reps in the Tank)",
            7.5: "(Possibly 4+ Reps in the Tank)",
            7.0: "(Possibly 5+ Reps in the Tank)"
        }

        rep_data.append({
            "rep": idx + 1,
            "time_sec": start / fps,
            "duration_sec": round(duration_sec, 2),
            "total_TUT": total_tut,
            "estimated_RPE": rpe,
            "estimated_RIR": rir_lookup.get(rpe, "Unknown")
        })

# Save rep data
with open(json_path, "w") as f:
    json.dump(rep_data, f, indent=4)

# Print summary
print("\n📊 Summary of Detected Reps:")
for rep in rep_data:
    print(f"Rep {rep['rep']}: Duration={rep['duration_sec']:.2f}s, Estimated RPE={rep['estimated_RPE']}, {rep['estimated_RIR']}, Total TUT={rep['total_TUT']:.2f}s")
if rep_data:
    print(f"Final Rep Estimated RPE: {rep_data[-1]['estimated_RPE']}")
