import torch
from .utils import *

def get_k_online_summary_update_index(summary, observation, k, threshold=None, dist_func="l2_dist", summary_dist_matrix=None, fill_first=False):
    """
    Alg. 1 from Girdhar et al. ICRA 2012 https://www.cim.mcgill.ca/~mrl/pubs/girdhar/icra2012.pdf
    Returns index to replace with new observation, threshold, the distance matrix for the summary, and the distance vector for the observation
    Fill_first means to fill the summary set without consideration of the score (function will still compute the score though)
    """
    sample_size = summary.shape[0]

    if sample_size < 1:
        return sample_size, None, None, sample_size.new_zeros((1,1)), sample_size.new_zeros((1,1))

    if threshold is None:
        threshold, summary_dist_matrix = get_mean_summary_score(summary, dist_func=dist_func, dist_matrix=summary_dist_matrix)

    if summary_dist_matrix is None:
        _, summary_dist_matrix = get_mean_summary_score(summary, dist_func=dist_func, dist_matrix=summary_dist_matrix)

    score, _, obs_dists = get_summary_score(summary, observation, dist_func=dist_func)

    # If fill first, and not yet k samples, just add
    if fill_first and sample_size < k:
        return sample_size, score, threshold, summary_dist_matrix, obs_dists

    # Do not add if score does not exceed threshold
    if score <= threshold:
        return -1, score, threshold, summary_dist_matrix, obs_dists

    # Simply add if summary size is smaller than k
    if score > threshold and sample_size < k:
        return sample_size, score, threshold, summary_dist_matrix, obs_dists

    # Add observation scores to the bottom row of the distance matrix
    dists_matrix = torch.cat((summary_dist_matrix, obs_dists.unsqueeze(0)), dim=0)

    # faster way to find the worst scoring one?
    S_inds = [k]
    Z_inds = list(range(k))

    for i in range(k-1):
        S = dists_matrix.index_select(0, summary.new_tensor(S_inds, dtype=int))
        Z_intersect_S = S.index_select(1, summary.new_tensor(Z_inds, dtype=int))

        min_s, min_ind = torch.min(Z_intersect_S, dim=0)
        new_ind = Z_inds[torch.argmax(min_s).item()]

        S_inds.append(new_ind)
        Z_inds.remove(new_ind)

        S_inds.sort()
        Z_inds.sort()

    replacement_ind = Z_inds[0]
    return replacement_ind, score, threshold, summary_dist_matrix, obs_dists