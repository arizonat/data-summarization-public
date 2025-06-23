import torch
from .utils import *

def knn(x, y, k, dist_func="norm_l2"):
    """
    Get the indices of the k nearest neighbors in y for each point in x
    """
    dists = get_pairwise_distances(x, y, dist_func=dist_func)
    _, indices = torch.topk(dists, k, largest=False, sorted=True)
    return indices

def nn_classify(x, targets, target_labels=None, dist_func="norm_l2"):
    """
    Get the label of the nearest point of x in target according to target_labels
    input: x: [[any dimension], C], target: [K, C], target_labels: [K]
    output: labels: [[any dimension]], dists: [[any dimension]]
    """

    if target_labels is None:
        target_labels = torch.arange(len(targets), device=x.device)

    if type(targets) == list:
        targets = torch.stack(targets)
    
    if type(target_labels) == list:
        target_labels = torch.tensor(target_labels, device=x.device)

    embed_dim = x.shape[-1]
    des_shape = x.shape[:-1]
    x = x.view(-1, embed_dim)
    dists = get_pairwise_distances(x, targets, dist_func=dist_func)
    # label_inds = torch.argmin(dists, dim=-1)
    label_dists, label_inds = torch.min(dists, dim=-1)

    labels = target_labels[label_inds]
    labels = labels.view(des_shape)
    label_dists = label_dists.view(des_shape)
    return labels, label_dists