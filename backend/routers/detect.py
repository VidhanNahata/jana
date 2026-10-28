from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from backend.routers.auth import get_current_user
from backend.models.user import User
from backend.services.pothole_detector import detect_potholes

router = APIRouter()

MAX_FRAME_BYTES = 5 * 1024 * 1024


@router.post("/pothole")
async def detect_pothole(
    frame: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    content = await frame.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty frame")
    if len(content) > MAX_FRAME_BYTES:
        raise HTTPException(status_code=413, detail="Frame too large")

    try:
        return await run_in_threadpool(detect_potholes, content)
    except OSError:
        raise HTTPException(status_code=400, detail="Invalid image")
