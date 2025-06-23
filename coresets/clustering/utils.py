import torch
import random
import numpy as np
from torch.nn.functional import normalize
from tqdm import tqdm

def set_seed(seed: int):
    """
    Set the seed for all common libraries to ensure reproducibility.
    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU
    # Ensure deterministic behavior in PyTorch (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

def get_device_memory(device="cuda:0"):
        dev = torch.device(device)
        free, total = torch.cuda.mem_get_info(dev)
        mem_used_gb = (total - free) / 1024 ** 3
        # print(mem_used_gb)
        return mem_used_gb

def pairwise_l2_distances(x, y=None, squared=False):
    '''
    This code is taken from: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    del x_norm, y_norm, y_t
    torch.cuda.empty_cache()

    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    dist = torch.clamp(dist, 0.0, float('inf'))
    if not squared:
        dist = torch.sqrt(dist)
    return dist

def pairwise_cosine_similarity(a, b):
    # Normalize the vectors
    a_norm = a / a.norm(dim=1, keepdim=True)
    b_norm = b / b.norm(dim=1, keepdim=True)
    similarity = torch.matmul(a_norm, b_norm.transpose(0, 1))
    
    return similarity

def get_pairwise_distances_chunked(x, y, chunk_size=1000, dist_func="norm_l2", device=None):
    """
    Compute pairwise distances in chunks to save memory
    inputs: x, y: [N, M] and [K, M]
    outputs: dists: [N, K]
    """
    orig_device = x.device

    n_samples_x = x.shape[0]
    n_samples_y = y.shape[0]
    
    dists = torch.zeros((n_samples_x, n_samples_y), device=orig_device)

    n_chunks = max(n_samples_x, n_samples_y) // chunk_size + (1 if max(n_samples_x, n_samples_y) % chunk_size > 0 else 0)

    x_start = 0
    for x_chunk in torch.chunk(x, n_chunks, dim=0):
        x_c = x_chunk.to(device)

        y_start = 0
        for y_chunk in torch.chunk(y, n_chunks, dim=0):
            y_c = y_chunk.to(device)
            chunk_dists = get_pairwise_distances(x_c, y_c, dist_func=dist_func, device=device)
            
            # Get the indices for the current chunk
            dists[x_start:chunk_dists.shape[0] + x_start, y_start:y_start + chunk_dists.shape[1]] = chunk_dists.to(orig_device)
            
            # Update the starting indices for the next chunk
            y_start = y_c.shape[0] if y_start is None else y_start + y_c.shape[0]
            
            # Reset the chunk tensors to free memory
            del y_c, chunk_dists
            if "cuda" in device:
                torch.cuda.empty_cache()

        x_start = x_c.shape[0] if x_start is None else x_start + x_c.shape[0]
        del x_c
        if "cuda" in device:
            torch.cuda.empty_cache()    
    return dists

def get_pairwise_distances(X, Y, dist_func="norm_l2", device=None):
    """
    inputs: X, Y: [N, M] and [K, M]
    outputs: dists: [N, K]
    """

    orig_device = X.device

    if device is None:
        device = X.device
    
    # Move tensors to the specified device
    X = X.to(device)
    Y = Y.to(device)

    if dist_func=="norm_l2":
        #dists = torch.cdist(normalize(X, dim=1), normalize(Y, dim=1))
        dists = pairwise_l2_distances(normalize(X, dim=1), normalize(Y, dim=1))
    elif dist_func=="cosine_similarity":
        dists = 1 - pairwise_cosine_similarity(X, Y)
    else:
        # dists_old = torch.cdist(X, Y)
        dists = pairwise_l2_distances(X, Y)

    if len(dists.shape) == 1:
        dists = dists.unsqueeze(1)

    return dists.to(orig_device)

def get_elementwise_surprise(data, summaries, dist_func="norm_l2", device=None):
    dists = get_pairwise_distances(data, summaries, dist_func=dist_func, device=device)
    element_surprise, element_surprise_inds = torch.min(dists, dim=1)
    return element_surprise, element_surprise_inds

def binary_search_continuous(func, target, x_min, x_max, tolerance=1e-2, max_iterations=100):
    """
    Binary search in continuous space to find x where func(x) = target

    Parameters:
    - func: The continuous function to search
    - target: The target value to find
    - x_min: Lower bound of search interval
    - x_max: Upper bound of search interval
    - tolerance: Acceptable error (when to stop iterating)
    - max_iterations: Maximum number of iterations

    Returns:
    - The approximate x value where func(x) = target, or None if not found
    """
    iterations = 0

    # Check if target is in range
    f_min = func(x_min)
    f_max = func(x_max)

    if (target < f_min and target < f_max) or (target > f_min and target > f_max):
        return None  # Target outside of interval

    with tqdm(total=max_iterations) as pbar:
        while x_max - x_min > tolerance and iterations < max_iterations:
            mid = (x_min + x_max) / 2
            f_mid = func(mid)
            
            if abs(f_mid - target) < tolerance:
                return mid  # Found target within tolerance
            
            if (f_mid < target and f_min < f_max) or (f_mid > target and f_min > f_max):
                x_min = mid
                f_min = f_mid
            else:
                x_max = mid
                f_max = f_mid
                
            iterations += 1
            pbar.update(1)

    # Return midpoint of final interval
    return (x_min + x_max) / 2