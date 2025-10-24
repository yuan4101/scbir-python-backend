import os
import ast
from typing import List, Optional

import cv2
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from skimage.feature import local_binary_pattern
from supabase import create_client, Client


# =========================
# Configuración
# =========================

SUPABASE_URL = os.getenv(
    "SUPABASE_URL",
    "https://oielbczfjirunzydccod.supabase.co"
)
SUPABASE_SERVICE_ROLE_KEY = os.getenv(
    "SUPABASE_SERVICE_ROLE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9pZWxiY3pmamlydW56eWRjY29kIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MTE0ODI1MSwiZXhwIjoyMDc2NzI0MjUxfQ.gOTR08x_7xqKjvsoeeqgesSB43cUFgEXShwSJ3DJ6jk"
)

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    print("⚠️  ADVERTENCIA: Faltan variables SUPABASE_URL o SUPABASE_SERVICE_ROLE_KEY")

supabase: Optional[Client] = None
try:
    if SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
        print("✅ Supabase inicializado correctamente")
except Exception as e:
    print(f"❌ No se pudo inicializar Supabase: {e}")
    supabase = None

IMAGE_SIZE = (256, 256)

app = FastAPI(title="CBIR HSV+LBP para carros", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Utilidades - Procesamiento de Imagen
# =========================

def load_image_from_bytes(content: bytes) -> np.ndarray:
    """Decodifica bytes y redimensiona imagen al tamaño estándar."""
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")
    return cv2.resize(img, IMAGE_SIZE)


def extraer_color_imagen(image: np.ndarray) -> np.ndarray:
    """Calcula histograma 3D HSV (8x8x8 bins = 512 dims)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def extraer_lbp_imagen(image_gray: np.ndarray) -> np.ndarray:
    """Calcula LBP uniforme (P=8, R=1) con histograma de 256 bins."""
    lbp = local_binary_pattern(image_gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def segmentar_con_grabcut(image: np.ndarray) -> np.ndarray:
    """Aplica segmentación GrabCut para aislar el objeto del fondo."""
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    h, w = image.shape[:2]
    rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
    
    try:
        cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        image_segmentada = image * mask2[:, :, np.newaxis]
        return image_segmentada
    except Exception as e:
        print(f"⚠️  GrabCut falló: {e}")
        return image


# =========================
# Salud
# =========================

@app.get("/health")
def health():
    """Endpoint de salud del servicio."""
    return {
        "status": "ok",
        "supabase": bool(supabase),
        "image_size": IMAGE_SIZE,
        "versions": ["v1", "v2"],
    }


# =========================
# Versión 1: HSV+LBP Estándar
# =========================

@app.post("/cbir/precompute/v1")
async def precompute_features_v1():
    """Precalcula vectores V1 (HSV+LBP) para todas las imágenes."""
    if supabase is None:
        raise HTTPException(500, "Supabase no inicializado")
    
    try:
        resp = supabase.table("carros").select("id, imagen").execute()
        rows = resp.data or []
        
        if not rows:
            return {"message": "No hay registros", "processed": 0}
        
        processed = 0
        errors = []
        
        for idx, r in enumerate(rows, 1):
            record_id = r.get("id")
            image_url = r.get("imagen")
            
            if not image_url:
                continue
            
            try:
                response = requests.get(image_url, timeout=15)
                response.raise_for_status()
                
                img = load_image_from_bytes(response.content)
                
                color_vec = extraer_color_imagen(img)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                texture_vec = extraer_lbp_imagen(gray)
                
                full_vector = np.concatenate([color_vec, texture_vec])
                vector_str = "[" + ",".join(map(str, full_vector.tolist())) + "]"
                
                supabase.table("carros").update({
                    "vector_caracteristicas_v1": vector_str
                }).eq("id", record_id).execute()
                
                processed += 1
                print(f"✅ V1 [{idx}/{len(rows)}] {record_id}")
                
            except Exception as e:
                errors.append({"id": record_id, "error": str(e)})
                print(f"❌ V1 [{idx}/{len(rows)}] {record_id}: {e}")
        
        return {
            "version": "v1",
            "message": "Procesamiento completado",
            "processed": processed,
            "total": len(rows),
            "errors": errors[:10]
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error en precálculo V1: {e}")


@app.post("/cbir/search/v1")
async def cbir_search_v1(
    file: Optional[UploadFile] = File(None, description="Imagen de consulta"),
    threshold: float = Form(0.3, description="Umbral de similitud [0-1]"),
    top_k: int = Form(12, description="Número de resultados"),
):
    """Búsqueda V1 usando similitud coseno."""
    if supabase is None:
        raise HTTPException(500, "Supabase no inicializado")
    
    if not file:
        raise HTTPException(400, "Falta parámetro 'file'")
    
    try:
        content = await file.read()
        qimg = load_image_from_bytes(content)
        
        q_color = extraer_color_imagen(qimg).reshape(1, -1)
        q_texture = extraer_lbp_imagen(cv2.cvtColor(qimg, cv2.COLOR_BGR2GRAY)).reshape(1, -1)
        
        resp = supabase.table("carros").select("id, imagen, vector_caracteristicas_v1").execute()
        rows = resp.data or []
        
        results: List[dict] = []
        
        for r in rows:
            vec = r.get("vector_caracteristicas_v1")
            
            if isinstance(vec, str):
                try:
                    vec = ast.literal_eval(vec)
                except Exception:
                    continue
            
            if not isinstance(vec, list):
                continue
            
            arr = np.array(vec, dtype=np.float32)
            if arr.ndim != 1 or arr.size < 768:
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
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        results = results[:max(1, min(top_k, 100))]
        
        return {
            "version": "v1",
            "carros": results,
            "total": len(results),
            "totalPages": 1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error en búsqueda V1: {e}")


# =========================
# Versión 2: GrabCut + Distancia Euclidiana
# =========================

@app.post("/cbir/precompute/v2")
async def precompute_features_v2():
    """Precalcula vectores V2 (GrabCut + HSV+LBP) para todas las imágenes."""
    if supabase is None:
        raise HTTPException(500, "Supabase no inicializado")
    
    try:
        resp = supabase.table("carros").select("id, imagen").execute()
        rows = resp.data or []
        
        if not rows:
            return {"message": "No hay registros", "processed": 0}
        
        processed = 0
        errors = []
        
        for idx, r in enumerate(rows, 1):
            record_id = r.get("id")
            image_url = r.get("imagen")
            
            if not image_url:
                continue
            
            try:
                response = requests.get(image_url, timeout=15)
                response.raise_for_status()
                
                img = load_image_from_bytes(response.content)
                img_segmentada = segmentar_con_grabcut(img)
                
                color_vec = extraer_color_imagen(img_segmentada)
                gray = cv2.cvtColor(img_segmentada, cv2.COLOR_BGR2GRAY)
                texture_vec = extraer_lbp_imagen(gray)
                
                full_vector = np.concatenate([color_vec, texture_vec])
                vector_str = "[" + ",".join(map(str, full_vector.tolist())) + "]"
                
                supabase.table("carros").update({
                    "vector_caracteristicas_v2": vector_str
                }).eq("id", record_id).execute()
                
                processed += 1
                print(f"✅ V2 [{idx}/{len(rows)}] {record_id}")
                
            except Exception as e:
                errors.append({"id": record_id, "error": str(e)})
                print(f"❌ V2 [{idx}/{len(rows)}] {record_id}: {e}")
        
        return {
            "version": "v2",
            "message": "Procesamiento completado",
            "processed": processed,
            "total": len(rows),
            "errors": errors[:10]
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error en precálculo V2: {e}")


@app.post("/cbir/search/v2")
async def cbir_search_v2(
    file: Optional[UploadFile] = File(None, description="Imagen de consulta"),
    threshold: float = Form(0.3, description="Umbral de similitud [0-1]"),
    top_k: int = Form(12, description="Número de resultados"),
):
    """Búsqueda V2 usando GrabCut + distancia euclidiana ponderada."""
    if supabase is None:
        raise HTTPException(500, "Supabase no inicializado")
    
    if not file:
        raise HTTPException(400, "Falta parámetro 'file'")
    
    try:
        content = await file.read()
        qimg = load_image_from_bytes(content)
        qimg_segmentada = segmentar_con_grabcut(qimg)
        
        q_color = extraer_color_imagen(qimg_segmentada).reshape(1, -1)
        q_texture = extraer_lbp_imagen(cv2.cvtColor(qimg_segmentada, cv2.COLOR_BGR2GRAY)).reshape(1, -1)
        
        resp = supabase.table("carros").select("id, imagen, vector_caracteristicas_v2").execute()
        rows = resp.data or []
        
        results: List[dict] = []
        
        for r in rows:
            vec = r.get("vector_caracteristicas_v2")
            
            if isinstance(vec, str):
                try:
                    vec = ast.literal_eval(vec)
                except Exception:
                    continue
            
            if not isinstance(vec, list):
                continue
            
            arr = np.array(vec, dtype=np.float32)
            if arr.ndim != 1 or arr.size < 768:
                continue
            
            color_vec = arr[:512].reshape(1, -1)
            texture_vec = arr[512:768].reshape(1, -1)
            
            dist_color = float(euclidean_distances(q_color, color_vec)[0][0])
            dist_texture = float(euclidean_distances(q_texture, texture_vec)[0][0])
            
            distancia_total = 0.7 * dist_color + 0.3 * dist_texture
            similarity = 1.0 / (1.0 + distancia_total)
            
            if similarity >= threshold:
                results.append({
                    "id": r.get("id"),
                    "imagen": r.get("imagen"),
                    "similarity": similarity,
                })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        results = results[:max(1, min(top_k, 100))]
        
        return {
            "version": "v2",
            "carros": results,
            "total": len(results),
            "totalPages": 1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error en búsqueda V2: {e}")
