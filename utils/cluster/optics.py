from collections import defaultdict
from sklearn.cluster import OPTICS
import torch

def cluster_with_optics(dm: torch.Tensor, min_samples=6, xi=0.001):
    """
    Clusters samples based on a pairwise distance matrix using OPTICS.

    Args:
        dm (torch.Tensor): Pairwise distance matrix (N x N).
        min_samples (int): Minimum number of samples in a cluster.
        xi (float): Determines how strict the cluster boundary detection is.

    Returns:
        dict: Each sublist contains indices of samples in the same cluster.
    """
    assert dm.ndim == 2 and dm.shape[0] == dm.shape[1], "dm must be a square 2D tensor"
    dm_np = dm.cpu().numpy()

    # Use OPTICS with precomputed distance matrix
    clustering = OPTICS(min_samples=min_samples, xi=xi, metric='precomputed')
    # clustering = OPTICS(min_samples=min_samples, xi=xi, metric='precomputed', cluster_method='dbscan')
    labels = clustering.fit_predict(dm_np)

    # Group sample indices by cluster label (excluding noise if any)
    cluster_dict = defaultdict(list)
    for idx, label in enumerate(labels):
        cluster_dict[label].append(idx)

    return cluster_dict