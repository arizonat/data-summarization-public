import torch
import clustering
import clustering.kcenters_robust_girdhar

class FixedSizeStreamingCoreset:
    def __init__(self, K, data:torch.tensor=None, cluster_algorithm=clustering.kcenters_robust_girdhar.kcover, device="cpu", dist_func="norm_l2"):
        """
        Example usage: 
            cl = lambda x, k: algorithms.kcenters_robust_girdhar.kcover(x, k, 0.5, dist_func="norm_l2", device=device, init_samples="random")
            cs = FixedSizeStreamingCoreset(K, cluster_algorithm=cl)
            cs.add(A)
            cs.add(B)
            print(cs.get_set())
        """
        self.K = K
        self.coreset = torch.tensor([], device=device)
        self.dist_func = dist_func
        self.cluster_algorithm = cluster_algorithm

        if data is not None:
            self.add(data, device=device)

    def __len__(self):
        return len(self.coreset)

    def __getitem__(self, idx):
        return self.coreset[idx]
    
    def get_set(self):
        return self.coreset

    def add(self, new_data:torch.tensor, device="cpu"):
        self.coreset = torch.cat((self.coreset, new_data), dim=0)

        # if len(self.features) > self.K:
        #     inds, self.features = kcenters_robust_index_chunked(self.features, self.K, delta=0.5, dist_func=self.dist_func, device=device, init_samples="random", stop_at_k=True, m_limit=None)

        # if len(self.features) > self.K:
        #     inds, self.features = kcover_index(self.features, self.K, 0.5, dist_func=self.dist_func, device=device, init_samples="random")

        if len(self.coreset) > self.K:
            inds, self.coreset = self.cluster_algorithm(self.coreset, self.K, 0.5, dist_func=self.dist_func, device=device, init_samples="random")
