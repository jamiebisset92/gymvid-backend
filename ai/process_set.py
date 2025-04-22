import os
import sys
import openai
import base64
import json
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Validate input
if len(sys.argv) < 2:
    print("Usage: python process_set.py path/to/video.mov [--coach]")
    sys.exit(1)

video_path = sys.argv[1]
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video not found: {video_path}")

run_coaching = "--coach" in sys.argv

# Step 1: Extract rep01 keyframes first for exercise ID
print("\n🖼️ Extracting keyframes...")
subprocess.run(["python3.10", "ai/extract_keyframes.py"])

# Step 2: Select 3 keyframes for exercise classification (from rep01 only)
keyframes_dir = "outputs/keyframes"
rep1_keyframes = ["rep01_start.jpg", "rep01_peak.jpg", "rep01_stop.jpg"]
valid_images = []

for filename in rep1_keyframes:
    path = os.path.join(keyframes_dir, filename)
    if not os.path.exists(path):
        print(f"❌ Keyframe not found: {filename}")
        continue
    with open(path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")
        valid_images.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
        })

# Step 3: Ask GPT to classify the exercise and estimate weight
print("\n📌 Identifying exercise and estimating weight...")
exercise = "Unknown"
confidence = 0
estimated_weight = "N/A"
visibility = "N/A"

if valid_images:
    response = openai.chat.completions.create(
        model="gpt-4o",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a fitness expert. Your job is to analyze the images from a workout set.\n"
                    "1. Identify the exercise being performed in 3-5 words.\n"
                    "2. Estimate the total weight being used based on visible plates or dumbbells.\n"
                    "3. Assume all Olympic barbells are 20kg unless clearly not present.\n"
                    "4. If only one side of a barbell or machine is visible, multiply it by 2 to calculate total.\n"
                    "5. If only one dumbbell is seen, use that weight. If two, multiply by 2.\n"
                    "6. For kettlebells, use color if visible.\n"
                    "7. Return a confidence score (0–100) on the exercise classification.\n"
                    "8. Return a confidence score (0–100) for how visible the weights are.\n"
                    "Return only JSON in this format: {\"exercise\": \"Barbell Deadlift\", \"confidence\": 92, \"weight_kg\": 100, \"weight_visibility\": 88}"
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Classify the exercise and estimate weight in kg. Return only JSON."},
                    *valid_images
                ]
            }
        ],
        max_tokens=200
    )

    try:
        content = response.choices[0].message.content.strip()
        print(f"\n🔎 Raw GPT Response:\n{content}")

        if content.startswith("```"):
            parts = content.split("```")
            content = next((p for p in parts if "{" in p and "}" in p), "{}").strip()

        parsed = json.loads(content)
        exercise = parsed.get("exercise", "Unknown")
        confidence = parsed.get("confidence", 0)
        estimated_weight = parsed.get("weight_kg", "N/A")
        visibility = parsed.get("weight_visibility", "N/A")
    except Exception as e:
        print("❌ Failed to parse exercise classification response.")
        print(str(e))

# Step 4: Display exercise result
print("\n📊 Exercise Prediction Result:")
if confidence < 85:
    print(f"Predicted: [{exercise}] — Confidence: {confidence}% (needs confirmation)")
else:
    print(f"Predicted: {exercise} — Confidence: {confidence}% ✅")
print(f"Estimated Weight: {estimated_weight} kg")
print(f"Weight Visibility Confidence: {visibility}%")

# Step 5: Run rep analysis with exercise context
print("\n📈 Running rep analysis...")
subprocess.run(["python3.10", "ai/analyze_video.py", video_path])

# Step 6: Run coaching feedback from ALL keyframes (optional)
if run_coaching:
    print("\n🧠 Generating coach feedback...")
    feedback_result = subprocess.run([
        "python3.10", "ai/coaching_feedback.py"
    ], capture_output=True, text=True)

    print("\n🤖 Coach Feedback:")
    feedback_output = feedback_result.stdout.strip()
    if feedback_output and not feedback_output.startswith("I'm sorry"):
        print(feedback_output)
    else:
        print("❌ No coach feedback returned. Please check the feedback script.")
else:
    print("\n🧠 Coach feedback skipped (no --coach flag provided).")

# ✅ Final summary
print("\n✅ Full set processing complete.")
