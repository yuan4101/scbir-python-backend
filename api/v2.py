import ast
from typing import List, Optional

import numpy as np
import requests
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from services.supabase_service import supabase_service
from features.image_processing import load_image_from_bytes, segmentar_con_grabcut
from features.feature_extraction import extract_combined_features
from features.similarity import calculate_weighted_euclidean

router = APIRouter(prefix="/cbir", tags=["CBIR V2"])

@router.post("/precompute/v2")
async def precompute_features_v2():
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
                img_segmentada = segmentar_con_grabcut(img)
                
                full_vector = extract_combined_features(img_segmentada)
                
                vector_str = "[" + ",".join(map(str, full_vector.tolist())) + "]"
                supabase_service.update_vector(record_id, "vector_caracteristicas_v2", vector_str)
                
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

@router.post("/search/v2")
async def cbir_search_v2(
    file: Optional[UploadFile] = File(None),
    threshold: float = Form(0.3),
    top_k: int = Form(12),
):
    if not file:
        raise HTTPException(400, "Falta parámetro 'file'")
    
    try:
        content = await file.read()
        qimg = load_image_from_bytes(content)
        qimg_segmentada = segmentar_con_grabcut(qimg)
        
        q_vector = extract_combined_features(qimg_segmentada)
        q_color = q_vector[:512].reshape(1, -1)
        q_texture = q_vector[512:768].reshape(1, -1)
        
        rows = supabase_service.get_carros_with_vector("vector_caracteristicas_v2")
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
            
            db_color = arr[:512].reshape(1, -1)
            db_texture = arr[512:768].reshape(1, -1)
            
            similarity = calculate_weighted_euclidean(q_color, q_texture, db_color, db_texture)
            
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
