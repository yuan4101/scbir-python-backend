from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import health, v1, v2

app = FastAPI(
    title="CBIR HSV+LBP para carros",
    version="2.0.0",
    description="Sistema de búsqueda de imágenes por contenido para vehículos"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(v1.router)
app.include_router(v2.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
