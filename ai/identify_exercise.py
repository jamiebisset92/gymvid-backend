import os
import sys
import openai
import base64
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Validate input
if len(sys.argv) < 2:
    print("Usage: python identify_exercise.py path/to/image1.jpg path/to/image2.jpg ...")
    sys.exit(1)

image_paths = sys.argv[1:]
valid_images = []

for path in image_paths:
    if not os.path.exists(path):
        print(f"❌ Image not found: {path}")
        continue
    with open(path, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")
        valid_images.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
        })

if not valid_images:
    print("❌ No valid images found.")
    sys.exit(1)

# Ask GPT to classify exercise from multiple frames
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a fitness expert who can recognize weightlifting and gym-based exercises from images. "
                "Based on the images provided, classify the exercise being performed. "
                "Return a short label (e.g., 'Barbell Back Squat') and an approximate confidence score between 0 and 100. "
                "Only output a JSON object like this: {\"exercise\": \"Barbell Back Squat\", \"confidence\": 93}"
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Classify the exercise shown across these images. Return only JSON."},
                *valid_images
            ]
        }
    ],
    max_tokens=100
)

# Extract and print result
content = response.choices[0].message.content.strip()
print(content)
