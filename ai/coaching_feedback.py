import os
import openai
import base64
from dotenv import load_dotenv

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Collect all keyframe paths
keyframes_dir = "outputs/keyframes"
if not os.path.exists(keyframes_dir):
    print("❌ No keyframes found.")
    exit()

image_files = sorted([
    os.path.join(keyframes_dir, f)
    for f in os.listdir(keyframes_dir)
    if f.endswith(".jpg")
])

if not image_files:
    print("❌ No .jpg keyframes in folder.")
    exit()

# Convert all images to base64
images_payload = []
for path in image_files:
    with open(path, "rb") as img_file:
        base64_img = base64.b64encode(img_file.read()).decode("utf-8")
        images_payload.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_img}"
            }
        })

# Send coaching prompt to GPT-4o
response = openai.chat.completions.create(
    model="gpt-4o",
    temperature=0.2,
    messages=[
        {
            "role": "system",
            "content": (
                "You are an expert lifting coach. Your goal is to review this lifter's technique based on several images of their set.\n\n"
                "Speak directly to the lifter, using 'your' instead of 'the'. Only mention issues that are **clearly visible and genuinely significant**.\n\n"
                "If a specific issue occurs in a particular rep (e.g., rep 2), point it out directly (IF there's an obvious inconsistency with one of the reps you observe).\n\n"
                "For each issue, provide:\n"
                "- What you noticed (starting with 'Your...')\n"
                "- A short coaching cue to help them improve\n\n"
                "Avoid nitpicking small details. If their form looks great, say so!\n\n"
                "Also include:\n"
                "- A motivational final note, always encouraging (avoid alarmist or fear-based language)\n"
                "- A technique score out of 10, where 9–10 is used for strong form with minor flaws\n\n"
                "Respond with this format:\n\n"
                "🏋️ Technique Feedback:\n"
                "- Issue: ...\n"
                "- Cue: ...\n\n"
                "🏁 Overall Notes:\n"
                "- ...\n\n"
                "🎯 Technique Score: X/10"
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please review the following keyframes for coaching feedback."},
                *images_payload
            ]
        }
    ],
    max_tokens=700
)

# Print GPT feedback
print(response.choices[0].message.content.strip())
