# Data summarization approaches and implementations
Library of clustering/core-set selection/data summarization algorithms implemented with pytorch
Developed for use in vision-based applications alongside DINO feature extractors

Batch algorithms:
- K-centers Greedy
- K-centers Robust Girdhar

Streaming algorithms:
- K-centers Streaming (Broken during refactor, need to fix)
- DINO detector
  
Example usage:

```
import torch
from clustering.kcenters_greedy import kcenters_greedy

K = 4
X = torch.rand(10,3)
summary = kcenters_greedy(X, K)

print("X: ", X)
print("selected indices: ", summary[0])
print("summary: ", summary[1])
```

Refer to simple_clustering_experiments.ipynb for additional clustering experiments

# Authors
Levi "Veevee" Cai (cail@mit.edu)
