import ast
from typing import List, Optional

import numpy as np
import requests
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from services.supabase_service import supabase_service
from features.image_processing import load_image_from_bytes, segmentar_con_rembg
from features.feature_extraction import extract_combined_features
from features.similarity import calcular_distancia_v3

router = APIRouter(prefix="/cbir", tags=["CBIR V3"])

@router.post("/precompute/v3")
async def precompute_features_v3():
    """Precalcula vectores V3 (rembg + HSV+LBP) para todas las imágenes."""
    try:
        rows = supabase_service.get_all_carros()
        
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
                img_segmentada = segmentar_con_rembg(img)
                
                full_vector = extract_combined_features(img_segmentada)
                
                vector_str = "[" + ",".join(map(str, full_vector.tolist())) + "]"
                supabase_service.update_vector(record_id, "vector_caracteristicas_v3", vector_str)
                
                processed += 1
                print(f"✅ V3 [{idx}/{len(rows)}] {record_id}")
                
            except Exception as e:
                errors.append({"id": record_id, "error": str(e)})
                print(f"❌ V3 [{idx}/{len(rows)}] {record_id}: {e}")
        
        return {
            "version": "v3",
            "message": "Procesamiento completado",
            "processed": processed,
            "total": len(rows),
            "errors": errors[:10]
        }
        
    except Exception as e:
        raise HTTPException(500, f"Error en precálculo V3: {e}")

@router.post("/search/v3")
async def cbir_search_v3(
    file: Optional[UploadFile] = File(None),
    threshold: float = Form(0.3),
    top_k: int = Form(12),
    metrica: str = Form('euclidean'),
    peso_color: float = Form(0.7),
):
    """
    Búsqueda V3 usando rembg + métrica configurable + pesos personalizables.
    
    Parámetros:
    -----------
    file : imagen de consulta
    threshold : umbral de similitud (0-1)
    top_k : número de resultados
    metrica : 'euclidean', 'manhattan', 'cosine', 'hamming'
    peso_color : peso del descriptor de color (0-1), textura = 1 - peso_color
    """
    if not file:
        raise HTTPException(400, "Falta parámetro 'file'")
    
    if metrica not in ['euclidean', 'manhattan', 'cosine', 'hamming']:
        raise HTTPException(400, f"Métrica '{metrica}' no válida")
    
    if not 0 <= peso_color <= 1:
        raise HTTPException(400, "peso_color debe estar entre 0 y 1")
    
    try:
        content = await file.read()
        qimg = load_image_from_bytes(content)
        qimg_segmentada = segmentar_con_rembg(qimg)
        
        q_vector = extract_combined_features(qimg_segmentada)
        q_color = q_vector[:512].reshape(1, -1)
        q_texture = q_vector[512:768].reshape(1, -1)
        
        rows = supabase_service.get_carros_with_vector("vector_caracteristicas_v3")
        results: List[dict] = []
        
        for r in rows:
            vec = r.get("vector_caracteristicas_v3")
            
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
            
            db_color = arr[:512].reshape(1, -1)
            db_texture = arr[512:768].reshape(1, -1)
            
            similarity = calcular_distancia_v3(
                q_color, q_texture, 
                db_color, db_texture,
                metrica=metrica,
                peso_color=peso_color
            )
            
            if similarity >= threshold:
                results.append({
                    "id": r.get("id"),
                    "imagen": r.get("imagen"),
                    "similarity": similarity,
                })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        results = results[:max(1, min(top_k, 100))]
        
        return {
            "version": "v3",
            "metrica": metrica,
            "peso_color": peso_color,
            "peso_textura": 1.0 - peso_color,
            "carros": results,
            "total": len(results),
            "totalPages": 1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error en búsqueda V3: {e}")

@router.post("/search/v3/metrics")
async def cbir_search_v3_with_metrics(
    file: Optional[UploadFile] = File(None),
    threshold: float = Form(0.3),
    top_k: int = Form(12),
    peso_color: float = Form(0.7),
):
    """
    Búsqueda V3 con análisis de todas las métricas.
    
    Calcula la similitud usando las 4 métricas disponibles y retorna:
    1. Resultados individuales por cada métrica
    2. Resultados combinados usando el promedio de todas las métricas
    
    Parámetros:
    -----------
    file : imagen de consulta
    threshold : umbral de similitud (0-1)
    top_k : número de resultados
    peso_color : peso del descriptor de color (0-1)
    
    Retorna:
    --------
    {
        "carros": [lista ordenada por promedio de métricas],
        "metrics_breakdown": {desglose por cada métrica},
        "total": número de resultados,
        "totalPages": 1
    }
    """
    if not file:
        raise HTTPException(400, "Falta parámetro 'file'")
    
    if not 0 <= peso_color <= 1:
        raise HTTPException(400, "peso_color debe estar entre 0 y 1")
    
    try:
        content = await file.read()
        qimg = load_image_from_bytes(content)
        qimg_segmentada = segmentar_con_rembg(qimg)
        
        q_vector = extract_combined_features(qimg_segmentada)
        q_color = q_vector[:512].reshape(1, -1)
        q_texture = q_vector[512:768].reshape(1, -1)
        
        rows = supabase_service.get_carros_with_vector("vector_caracteristicas_v3")
        
        metricas_disponibles = ['euclidean', 'manhattan', 'cosine', 'hamming']
        results_by_metric = {metrica: [] for metrica in metricas_disponibles}
        combined_results = {}
        
        for r in rows:
            vec = r.get("vector_caracteristicas_v3")
            
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
            
            db_color = arr[:512].reshape(1, -1)
            db_texture = arr[512:768].reshape(1, -1)
            
            carro_id = r.get("id")
            carro_imagen = r.get("imagen")
            similarities = {}
            
            for metrica in metricas_disponibles:
                similarity = calcular_distancia_v3(
                    q_color, q_texture, 
                    db_color, db_texture,
                    metrica=metrica,
                    peso_color=peso_color
                )
                
                similarities[metrica] = similarity
                
                results_by_metric[metrica].append({
                    "id": carro_id,
                    "imagen": carro_imagen,
                    "similarity": similarity,
                })
            
            avg_similarity = sum(similarities.values()) / len(similarities)
            
            if avg_similarity >= threshold:
                combined_results[carro_id] = {
                    "id": carro_id,
                    "imagen": carro_imagen,
                    "similarity_avg": avg_similarity,
                    "metrics": similarities
                }
        
        for metrica in metricas_disponibles:
            results_by_metric[metrica].sort(key=lambda x: x["similarity"], reverse=True)
            results_by_metric[metrica] = results_by_metric[metrica][:top_k]
        
        final_results = sorted(
            combined_results.values(), 
            key=lambda x: x["similarity_avg"], 
            reverse=True
        )[:max(1, min(top_k, 100))]
        
        return {
            "version": "v3",
            "aggregation": "average",
            "peso_color": peso_color,
            "peso_textura": 1.0 - peso_color,
            "carros": final_results,
            "metrics_breakdown": results_by_metric,
            "total": len(final_results),
            "totalPages": 1
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error en búsqueda V3 con métricas: {e}")
