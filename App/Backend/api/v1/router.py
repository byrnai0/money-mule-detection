from fastapi import APIRouter

from api.v1.live_detection import router as live_detection_router

api_router = APIRouter()
api_router.include_router(live_detection_router, prefix="/analysis", tags=["analysis"])
