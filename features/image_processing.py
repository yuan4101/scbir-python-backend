import cv2
import numpy as np
from PIL import Image
from rembg import remove
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

def segmentar_con_rembg(image: np.ndarray) -> np.ndarray:
    """
    Aplica segmentación basada en Deep Learning usando rembg (U²-Net).
    Elimina el fondo completamente, dejando solo el vehículo sobre fondo negro.
    
    Ventajas sobre GrabCut:
    - Basado en red neuronal U²-Net pre-entrenada
    - Segmentación completamente automática
    - Bordes precisos y limpios
    - Elimina 100% del fondo
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    output_pil = remove(pil_image)
    output_np = np.array(output_pil)
    
    if output_np.shape[2] == 4:
        rgb = output_np[:, :, :3]
        alpha = output_np[:, :, 3]
        
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask = (alpha > 10).astype(np.uint8)
        image_segmentada = bgr * mask[:, :, np.newaxis]
        
        return image_segmentada
    else:
        return cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
