from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import shutil
import subprocess

app = FastAPI()

@app.post("/process_set")
async def process_set(
    video: UploadFile = File(...),
    coaching: bool = Form(False)
):
    # Save uploaded file
    save_path = f"temp_uploads/{video.filename}"
    os.makedirs("temp_uploads", exist_ok=True)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Build command
    cmd = ["python3.10", "ai/process_set.py", save_path]
    if coaching:
        cmd.append("--coach")

    try:
        # Run script and capture output
        result = subprocess.run(cmd, capture_output=True, text=True)

        return JSONResponse({
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
