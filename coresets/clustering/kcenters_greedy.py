import torch
from .utils import *

def kcenters_greedy(data, k, dist_func="norm_l2", device=None, init_samples="random"):
    """
    K-centers or greedy extremum summary algorithm
    
    Args:
        data (torch.Tensor): Input data of shape (n_samples, n_features)
        k (int): Number of centers
        device (torch.device, optional): Device to run computation on
    
    Returns:
        center_indices (torch.Tensor): Indices of selected centers
        centers (torch.Tensor): Selected centers
    """
    if device is None:
        device = data.device
    
    data = data.to(device)
    n_samples = len(data)

    if init_samples == "random":
        init_ind = torch.randint(0, n_samples, (1,), device=device).item()
    elif init_samples == "last":
        init_ind = n_samples - 1
    elif init_samples == "first":
        init_ind = 0
    
    # Initialize first center
    center_indices = torch.tensor([init_ind], device=device)
    center_points = data[center_indices]
    
    # Keep track of minimum distances to closest center for each point
    min_distances = get_pairwise_distances(data, center_points, dist_func=dist_func, device=device).squeeze(1)

    # Select centers
    while len(center_indices) < k:
        # Choose furthest point as next center
        new_center_idx = torch.argmax(min_distances)
        center_indices = torch.cat([center_indices, new_center_idx.unsqueeze(0)])
        new_center = data[new_center_idx].unsqueeze(0)
        
        # Update minimum distances
        new_distances = get_pairwise_distances(data, new_center, dist_func=dist_func, device=device).squeeze(1)
        min_distances = torch.minimum(min_distances, new_distances)
    
    # Compute final assignments
    center_points = data[center_indices]
    
    return center_indices, center_points