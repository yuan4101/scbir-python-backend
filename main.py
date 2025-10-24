import os
import ast
from typing import List, Optional

import cv2
import numpy as np
#import pandas as pd  # opcional, no es requerido pero útil si luego quieres DataFrame
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import local_binary_pattern
from supabase import create_client, Client  # supabase-py v2

# =========================
# Configuración y clientes
# =========================

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://oielbczfjirunzydccod.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9pZWxiY3pmamlydW56eWRjY29kIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjExNDgyNTEsImV4cCI6MjA3NjcyNDI1MX0.zsHFZhc4pUFUEb8ialtZzSA1rSXTlVHhlS1yFTB0PdE")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    # Arranca igual para healthcheck; pero /cbir/search fallará con 500 hasta que definas las vars
    print("ADVERTENCIA: Faltan variables SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY")

supabase: Optional[Client] = None
try:
    if SUPABASE_URL and SUPABASE_ANON_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
except Exception as e:
    print(f"No se pudo inicializar Supabase: {e}")
    supabase = None

IMAGE_SIZE = (256, 256)

app = FastAPI(title="CBIR HSV+LBP para carros", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción, limita a tu dominio Frontend si llamas directo
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Utilidades de imagen
# =========================

def extraer_color_imagen(image: np.ndarray) -> np.ndarray:
    """
    Histograma 3D HSV con bins 8x8x8 => 512 dim, normalizado.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extraer_lbp_imagen(image_gray: np.ndarray) -> np.ndarray:
    """
    LBP uniforme con P=8, R=1; histograma 256 bins normalizado.
    """
    lbp = local_binary_pattern(image_gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def load_image_from_bytes(content: bytes) -> np.ndarray:
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")
    return cv2.resize(img, IMAGE_SIZE)

# =========================
# Salud
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "supabase": bool(supabase),
        "image_size": IMAGE_SIZE,
        "model": "HSV(512)+LBP(256)",
    }

# =========================
# Búsqueda CBIR contra Supabase
# =========================

@app.post("/cbir/search")
async def cbir_search(
    file: Optional[UploadFile] = File(None, description="Imagen de consulta"),
    threshold: float = Form(0.3, description="Umbral de similitud [0-1]"),
    top_k: int = Form(12, description="Número de resultados"),
):
    """
    Recibe una imagen, calcula HSV+LBP de la query, lee todos los registros de Supabase (id, imagen, vector_caracteristicas),
    calcula similitud coseno por segmentos y devuelve top_k.
    """
    if supabase is None:
        raise HTTPException(500, "Supabase no inicializado. Verifica variables SUPABASE_URL y SUPABASE_SERVICE_ROLE_KEY")

    if not file:
        raise HTTPException(400, "Envía 'file' en multipart/form-data")

    try:
        # 1) Vector de la query
        content = await file.read()
        qimg = load_image_from_bytes(content)
        q_color = extraer_color_imagen(qimg).reshape(1, -1)  # 512
        q_texture = extraer_lbp_imagen(cv2.cvtColor(qimg, cv2.COLOR_BGR2GRAY)).reshape(1, -1)  # 256

        # 2) Leer de Supabase solo las columnas necesarias
        # Nota: si la tabla es grande, considera paginar o usar filtros.
        resp = supabase.table("carros").select("id, imagen, vector_caracteristicas").execute()
        rows = resp.data or []

        results: List[dict] = []
        for r in rows:
            vec = r.get("vector_caracteristicas")

            # El campo puede venir como list[float] (si es json/array) o como str (si lo guardaste serializado).
            if isinstance(vec, str):
                try:
                    vec = ast.literal_eval(vec)
                except Exception:
                    continue
            if not isinstance(vec, list):
                continue

            arr = np.array(vec, dtype=np.float32)
            if arr.ndim != 1 or arr.size < 768:
                # Esperamos 512 HSV + 256 LBP = 768; ajusta si tu esquema difiere.
                continue

            hsv = arr[:512].reshape(1, -1)
            lbp = arr[512:768].reshape(1, -1)

            sim_color = float(cosine_similarity(q_color, hsv)[0][0])
            sim_texture = float(cosine_similarity(q_texture, lbp)[0][0])
            sim = (sim_color + sim_texture) / 2.0

            if sim >= threshold:
                results.append({
                    "id": r.get("id"),
                    "imagen": r.get("imagen"),
                    "similarity": sim,
                })

        # 3) Ordenar y truncar
        results.sort(key=lambda x: x["similarity"], reverse=True)
        results = results[: max(1, min(top_k, 100))]  # limita hard a 100

        # 4) Respuesta compatible con tu UI
        return {
            "carros": results,
            "total": len(results),
            "totalPages": 1
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"CBIR error: {e}")
