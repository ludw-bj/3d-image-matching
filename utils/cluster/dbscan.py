import numpy as np
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import torch

def cluster_with_DBSCAN(dm: torch.Tensor, min_samples=3):
    """
    Automatically estimates eps using the k-distance method and runs DBSCAN to cluster samples based on their Euclidean distance matrix

    Args:
        dm (torch.Tensor): Pairwise distance matrix (N x N).
        min_samples (int): Minimum number of samples to form a dense region.

    Returns:
        dict(idx, list): Each sublist contains indices of samples in the same cluster.
    """
    assert dm.ndim == 2 and dm.shape[0] == dm.shape[1], "dm must be a square 2D tensor"
    
    # Convert to numpy array for DBSCAN
    dm_np = dm.cpu().numpy()

    # Step 1: k-distance for each point
    nbrs = NearestNeighbors(n_neighbors=min_samples, metric='precomputed')
    nbrs.fit(dm)
    distances, _ = nbrs.kneighbors(dm)
    k_distances = distances[:, -1]  # distance to min_samples-th neighbor
    k_distances_sorted = np.sort(k_distances)

    # Step 2: elbow detection via maximum curvature
    diffs = np.diff(k_distances_sorted)
    diff_ratios = diffs[1:] / (diffs[:-1] + 1e-6)  # ratio of successive diffs
    elbow_idx = np.argmax(diff_ratios) + 1  # shift by 1 due to diff
    best_eps = k_distances_sorted[elbow_idx]
    
    # Use DBSCAN with precomputed distance matrix
    clustering = DBSCAN(eps=best_eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(dm_np)
    
    # Group sample indices by cluster label (excluding noise if any)
    cluster_dict = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_dict[label].append(idx)

    return cluster_dict