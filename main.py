import os
import shutil
import uuid
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from face import FaceAnalysisError, analyze_face


# =========================================================
# FRAMEANA - STAGE 1 BACKEND
# Stable Face Analysis only
# No 2D overlay, no CAD, no STL
# =========================================================


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="FRAMEANA Face Analysis API",
    description="Stage 1 backend: upload a face image and return stable facial measurements.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _validate_image_file(file: UploadFile) -> str:
    filename = file.filename or ""
    _, extension = os.path.splitext(filename.lower())

    if extension not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed image types: {allowed}",
        )

    return extension


@app.get("/")
def health_check() -> Dict[str, str]:
    return {
        "status": "running",
        "message": "FRAMEANA Stage 1 Face Analysis API is running.",
    }


@app.post("/analyze-face")
async def analyze_face_endpoint(file: UploadFile = File(...)) -> Dict[str, object]:
    """
    Upload a face image and return stable facial measurements.

    Measurements are in pixels.
    """
    extension = _validate_image_file(file)
    image_id = str(uuid.uuid4())
    image_path = os.path.join(UPLOAD_DIR, f"{image_id}{extension}")

    try:
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = analyze_face(image_path)
        measurements = result["measurements"]

        return {
            "status": "success",
            "unit": "pixels",
            "face_width": measurements["face_width"],
            "eye_distance": measurements["eye_distance"],
            "nose_width": measurements["nose_width"],
            "head_angle": measurements["head_angle"],
            "center_x": measurements["center_x"],
            "center_y": measurements["center_y"],
            "debug": result["debug"],
        }

    except FaceAnalysisError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected server error: {str(error)}",
        ) from error

    finally:
        await file.close()
        if os.path.exists(image_path):
            os.remove(image_path)
