import numpy as np
from collections import defaultdict
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import inconsistent
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import squareform
from collections import Counter
import torch

def auto_cutoff_inconsistency(Z, depth=2, threshold_ratio=0.5):
    """
    Estimate cut-off based on inconsistency in the linkage matrix.

    Args:
        Z: Linkage matrix.
        depth: Depth parameter for `inconsistent`.
        threshold_ratio: Controls sensitivity (larger → fewer clusters).
    
    Returns:
        float: Suggested cutoff distance.
    """
    incons = inconsistent(Z, d=depth)
    mean_last = incons[-1, 0]
    std_last = incons[-1, 1]
    cutoff = mean_last + threshold_ratio * std_last
    return cutoff

def auto_cutoff_elbow(Z):
    """
    Choose cut-off based on the largest gap between successive merge distances.
    
    Returns:
        float: Estimated cutoff.
    """
    distances = Z[:, 2]
    diffs = np.diff(distances)
    idx = np.argmax(diffs)
    return (distances[idx] + distances[idx+1]) / 2

def auto_cutoff_silhouette(condensed_dm, Z, min_clusters=2, max_clusters=10, steps=10, min_cluster_size=3):
    """
    Automatically choose cutoff that maximizes silhouette score, ignoring small clusters.

    Args:
        condensed_dm: 1D condensed distance matrix.
        Z: Linkage matrix.
        min_clusters: Minimum number of large clusters required.
        max_clusters: Maximum number of large clusters allowed.
        steps: Number of thresholds to test.
        min_cluster_size: Minimum number of samples in a cluster to count it as valid.

    Returns:
        float: Optimal cutoff value.
    """
    dm_square = squareform(condensed_dm)
    best_score = -1
    best_cutoff = auto_cutoff_elbow(Z)
    distances = Z[:, 2]

    for t in np.linspace(min(distances), max(distances), steps):
        labels = fcluster(Z, t=t, criterion='distance')
        
        # Count cluster sizes
        label_counts = Counter(labels)
        
        # Get labels of valid clusters (with enough samples)
        valid_cluster_labels = {label for label, count in label_counts.items() if count >= min_cluster_size}
        
        # Mask samples belonging to valid clusters
        mask = [label in valid_cluster_labels for label in labels]
        filtered_labels = [label for i, label in enumerate(labels) if mask[i]]
        filtered_dm = dm_square[np.ix_(mask, mask)]

        n_valid_clusters = len(set(filtered_labels))

        # Need at least 2 valid clusters for silhouette score to work
        if n_valid_clusters < min_clusters or n_valid_clusters > max_clusters:
            continue

        # score = silhouette_score(filtered_dm, filtered_labels, metric='precomputed')
        score = silhouette_score(dm_square, labels, metric='precomputed')

        if score > best_score:
            best_score = score
            best_cutoff = t

    return best_cutoff

def cluster_with_hierarchical(dm: torch.Tensor):
    """
    Hierarchical clustering based on a pairwise distance matrix

    Args:
        dm (torch.Tensor): Pairwise distance matrix (N x N).
        cut_off (float): Threshold to cut the dendrogram for flat clusters.

    Returns:
        dict: Dictionary where each key corresponds to a cluster label and 
              the value is a list of sample indices in that cluster.
    """
    assert dm.ndim == 2 and dm.shape[0] == dm.shape[1], "dm must be a square 2D tensor"

    # Convert to numpy and flatten upper triangular part (condensed distance matrix)
    dm_np = dm.cpu().numpy()
    condensed_dm = dm_np[np.triu_indices(len(dm_np), k=1)]

    # Hierarchical clustering with 'average' linkage
    Z = linkage(condensed_dm, method='average')
    # cut_off = auto_cutoff_inconsistency(Z)
    # cut_off = auto_cutoff_elbow(Z)
    cut_off = auto_cutoff_silhouette(condensed_dm, Z)

    # Form flat clusters with a threshold on the cophenetic distance
    labels = fcluster(Z, t=cut_off, criterion='distance')

    # Group sample indices by cluster label (excluding noise if any)
    cluster_dict = defaultdict(list)
    for idx, label in enumerate(labels):
        # print(f'{idx}:{label}')
        cluster_dict[label].append(idx)

    return cluster_dict