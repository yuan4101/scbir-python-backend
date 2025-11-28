from fastapi import FastAPI
from .health import router as health_router
from .v1 import router as v1_router
from .v2 import router as v2_router
from .v3 import router as v3_router

def register_routers(app: FastAPI):
    """Registra todos los routers de la API."""
    app.include_router(health_router)
    app.include_router(v1_router)
    app.include_router(v2_router)
    app.include_router(v3_router)

__all__ = ["register_routers"]
