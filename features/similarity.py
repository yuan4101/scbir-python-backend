import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def calculate_cosine_similarity(query_color: np.ndarray, query_texture: np.ndarray,
                                db_color: np.ndarray, db_texture: np.ndarray) -> float:
    sim_color = float(cosine_similarity(query_color, db_color)[0][0])
    sim_texture = float(cosine_similarity(query_texture, db_texture)[0][0])
    return (sim_color + sim_texture) / 2.0

def calculate_weighted_euclidean(query_color: np.ndarray, query_texture: np.ndarray,
                                 db_color: np.ndarray, db_texture: np.ndarray,
                                 color_weight: float = 0.7) -> float:
    dist_color = float(euclidean_distances(query_color, db_color)[0][0])
    dist_texture = float(euclidean_distances(query_texture, db_texture)[0][0])
    
    texture_weight = 1.0 - color_weight
    distancia_total = color_weight * dist_color + texture_weight * dist_texture
    
    return 1.0 / (1.0 + distancia_total)

def distancia_euclidiana(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = v1.flatten()
    v2 = v2.flatten()
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))

def distancia_manhattan(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = v1.flatten()
    v2 = v2.flatten()
    return float(np.sum(np.abs(v1 - v2)))

def distancia_coseno(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = v1.flatten()
    v2 = v2.flatten()
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 1.0
    
    cosine_sim = dot_product / (norm_v1 * norm_v2)
    return 1.0 - cosine_sim

def distancia_hamming(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = v1.flatten()
    v2 = v2.flatten()
    
    v1_binario = (v1 > np.mean(v1)).astype(int)
    v2_binario = (v2 > np.mean(v2)).astype(int)
    
    diferencias = np.sum(v1_binario != v2_binario)
    return float(diferencias / len(v1))

def calcular_distancia_v3(query_color: np.ndarray, query_texture: np.ndarray,
                          db_color: np.ndarray, db_texture: np.ndarray,
                          metrica: str = 'euclidean',
                          peso_color: float = 0.7) -> float:
    """
    Calcula similitud usando métrica y pesos configurables (V3).
    
    Parámetros:
    -----------
    query_color, query_texture : vectores de consulta
    db_color, db_texture : vectores de base de datos
    metrica : 'euclidean', 'manhattan', 'cosine', 'hamming'
    peso_color : peso del color (0.0-1.0), textura = 1 - peso_color
    
    Retorna:
    --------
    float : similitud (0-1, mayor = más similar)
    """
    metricas = {
        'euclidean': distancia_euclidiana,
        'manhattan': distancia_manhattan,
        'cosine': distancia_coseno,
        'hamming': distancia_hamming
    }
    
    if metrica not in metricas:
        raise ValueError(f"Métrica '{metrica}' no soportada. Opciones: {list(metricas.keys())}")
    
    funcion_metrica = metricas[metrica]
    peso_textura = 1.0 - peso_color
    
    dist_color = funcion_metrica(query_color, db_color)
    dist_texture = funcion_metrica(query_texture, db_texture)
    
    distancia_total = peso_color * dist_color + peso_textura * dist_texture
    similarity = 1.0 / (1.0 + distancia_total)
    
    return similarity
