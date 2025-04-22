import cv2
import os
import json

# ---- CONFIG ----
video_path = "Jamie_Deadlift.mov"
rep_data_path = "outputs/Jamie_Deadlift_rep_data.json"
output_dir = "outputs/keyframes"
os.makedirs(output_dir, exist_ok=True)

# ---- LOAD REP DATA ----
with open(rep_data_path, "r") as f:
    rep_data = json.load(f)

# ---- VIDEO SETUP ----
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ---- FRAME EXTRACTION ----
def save_frame(frame_number, label, rep_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if ret:
        filename = f"rep{rep_number:02d}_{label}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)
        print(f"✅ Saved: {filepath}")
    else:
        print(f"❌ Failed to read frame {frame_number}")

# ---- MAIN LOOP ----
for rep in rep_data:
    rep_num = rep["rep"]
    start_frame = int(rep["time_sec"] * fps)
    peak_frame = int((rep["time_sec"] + rep["duration_sec"]) * fps)
    stop_frame = int((rep["time_sec"] + rep["duration_sec"] + (rep["duration_sec"] * 1.2)) * fps)

    save_frame(start_frame, "start", rep_num)
    save_frame(peak_frame, "peak", rep_num)
    save_frame(stop_frame, "stop", rep_num)

cap.release()
print("🎉 Keyframe extraction complete.")
