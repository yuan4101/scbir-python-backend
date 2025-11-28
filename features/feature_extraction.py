import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extraer_color_imagen(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extraer_lbp_imagen(image_gray: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(image_gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_combined_features(image: np.ndarray) -> np.ndarray:
    color_vec = extraer_color_imagen(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_vec = extraer_lbp_imagen(gray)
    return np.concatenate([color_vec, texture_vec])
