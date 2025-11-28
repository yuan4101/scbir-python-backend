import cv2
import numpy as np
from config.settings import settings

def load_image_from_bytes(content: bytes) -> np.ndarray:
    arr = np.frombuffer(content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen")
    return cv2.resize(img, settings.IMAGE_SIZE)

def segmentar_con_grabcut(image: np.ndarray) -> np.ndarray:
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
