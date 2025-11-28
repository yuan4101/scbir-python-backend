from fastapi import APIRouter
from config.settings import settings

router = APIRouter()

@router.get("/health")
def health():
    return {
        "status": "ok",
        "image_size": settings.IMAGE_SIZE,
        "versions": ["v1", "v2"],
    }
