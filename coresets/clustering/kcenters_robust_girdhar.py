import torch
from .utils import *
from tqdm import tqdm

def kcover(data, k, gamma, dist_func="norm_l2", device=None, init_samples="smallest_mean"):
    # todo: just directly return the solutions, don't keep recomputing them
    print("Initializing...")
    dists = get_pairwise_distances(data, data, dist_func=dist_func, device=device)

    func = lambda x: len(gamma_kcenters(data, k, gamma=gamma, thresh=x, dists=dists, dist_func=dist_func, device=device, init_samples=init_samples, stop_at_k=False, silence_tqdm=True)[0])

    x_min = torch.min(dists)
    x_max = torch.max(dists)
    print("Searching for threshold...")
    thresh = binary_search_continuous(func, k, x_min, x_max, tolerance=1e-2, max_iterations=100)
    return gamma_kcenters(data, k, gamma=gamma, thresh=thresh, dists=dists, dist_func=dist_func, device=device, init_samples=init_samples)

def gamma_kcenters(data, k, gamma=1, thresh=0.9, dists=None, dist_func="norm_l2", device=None, init_samples="smallest_mean", stop_at_k=False, silence_tqdm=False):
    """
    Outlier K-center algorithm
    Based on "Offline Navigation Summaries" Girdhar et al. ICRA 2011
    dists = precomputed dists matrix, if not provided, will be computed
    """
    if device is None:
        device = data.device
    
    n_samples = len(data)

    Z = data.clone().to(device)
    Z_inds = torch.arange(n_samples, device=device)

    # Memoize
    if dists is None:
        all_dists = get_pairwise_distances(Z, Z, dist_func=dist_func, device=device)
    else:
        all_dists = dists.to(device)

    if init_samples == "random":
        init_ind = torch.randint(0, n_samples, (1,), device=device).item()
    elif init_samples == "last":
        init_ind = n_samples - 1
    elif init_samples == "first":
        init_ind = 0
    elif init_samples == "smallest_mean":
        # dists = get_pairwise_distances(Z, Z, dist_func=dist_func, device=device)
        mean_dists = torch.mean(all_dists, dim=0)
        init_ind = torch.argmin(mean_dists)
    
    # Initialize first center
    center_indices = torch.tensor([init_ind], device=device)
    center_points = data[center_indices:center_indices+1,...].clone().to(device) # assumes just 1 center index
    mask = torch.ones(Z.shape[0], device=device, dtype=torch.bool)
    mask[init_ind] = False
    Z = Z[mask,...]
    Z_inds = Z_inds[mask]

    # Compute summary coverage ratio: |C(S|T)| / |Z|
    csit = get_pairwise_distances(center_points, Z) < thresh
    csit.cpu()
    cst = torch.any(csit, dim=0)
    cst_z = torch.sum(cst) / Z.shape[0]

    with tqdm(total=k, disable=silence_tqdm) as pbar:
        while cst_z < gamma:
            if stop_at_k and len(center_points) == k:
                break

            # Compute distances and threholds for current centers: C(S_i|eps_T) 
            # csit = get_pairwise_distances(center_points, Z, dist_func=dist_func, device=device) < thresh
            csit = all_dists[center_indices[:,None], Z_inds] < thresh

            # Compute coverage set for current centers:  C(S|eps_T)
            cst = torch.any(csit, dim=0)
            cst = cst.repeat(Z.shape[0], 1) # broadcasting for next steps

            # Compute active set distances: C(Z_i|eps_T)
            # czit = get_pairwise_distances(Z, Z, dist_func=dist_func, device=device) < thresh #todo: we can re-use this
            czit = all_dists[Z_inds[:,None],Z_inds] < thresh

            # Compute coverage sets including and excluding active set points: |C(Z_iUS|eps_T)\C(S|eps_T)|
            czust = torch.logical_or(czit, cst) # union
            czust_minus_cst = torch.logical_xor(czust, cst) # delete

            # Greedily select active point with largest coverage: argmax z_i and add to center points
            z_max_ind = torch.argmax(czust_minus_cst.sum(dim=1))
            center_points = torch.concat((center_points, Z[z_max_ind:z_max_ind+1,...].clone()))
            center_indices = torch.concat((center_indices, Z_inds[z_max_ind:z_max_ind+1,...].clone()))

            # Remove selected center from active set: Z \ z_max
            mask = torch.ones(Z.shape[0], device=device, dtype=torch.bool)
            mask[z_max_ind] = False
            Z = Z[mask,...]
            Z_inds = Z_inds[mask]
            
            # Compute summary coverage ratio: |C(S|T)| / |Z|
            # csit = get_pairwise_distances(center_points, Z) < thresh
            csit = all_dists[center_indices[:,None], Z_inds] < thresh

            cst = torch.any(csit, dim=0)
            cst_z = torch.sum(cst) / Z.shape[0]

            torch.cuda.empty_cache()
            pbar.update(1)

    return center_indices, center_points
