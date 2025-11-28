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
