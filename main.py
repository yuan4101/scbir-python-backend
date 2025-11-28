from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import register_routers

app = FastAPI(
    title="CBIR HSV+LBP para carros",
    version="3.0.0",
    description="Sistema de búsqueda de imágenes por contenido para vehículos"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

register_routers(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
