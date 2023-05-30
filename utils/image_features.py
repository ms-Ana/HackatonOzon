from sklearn.metrics import pairwise_distances
import numpy as np
from typing import List
from scipy.spatial.distance import cosine, euclidean

def get_pic_features(main_pic_embeddings_1: np.ndarray,
                     main_pic_embeddings_2: np.ndarray,
                     metric:str,
                     percentiles: List[int]) -> List[float]:
    """Calculate distances percentiles for 
    pairwise pic distances. Percentiles are useful 
    when product has several pictures.
    """
    
    if main_pic_embeddings_1 is not None and main_pic_embeddings_2 is not None:
        main_pic_embeddings_1 = np.array([x for x in main_pic_embeddings_1])
        main_pic_embeddings_2 = np.array([x for x in main_pic_embeddings_2])
        
        dist_m = pairwise_distances(
            main_pic_embeddings_1, main_pic_embeddings_2,
            metric=metric
        )
    else:
        dist_m = np.array([[-1]])

    pair_features = []
    pair_features += np.percentile(dist_m, percentiles).tolist()

    return pair_features

def get_pic_features_mean(main_pic_embeddings_1: np.ndarray,
                     main_pic_embeddings_2: np.ndarray,
                     metric: str = "euclidean") -> float:
    if main_pic_embeddings_1 is not None and main_pic_embeddings_2 is not None:
        main_pic_embeddings_1 = np.mean(np.array([x for x in main_pic_embeddings_1]), axis=0)
        main_pic_embeddings_2 = np.mean(np.array([x for x in main_pic_embeddings_2]), axis=0)
        if metric == "euclidean":
            return euclidean(main_pic_embeddings_1, main_pic_embeddings_2)
        else:
            return cosine(main_pic_embeddings_1, main_pic_embeddings_2)
    return -1.0
             
    