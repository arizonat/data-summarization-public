"""
DINO utilities

Written by Levi Cai (cail@mit.edu)
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

from typing import Union, List, Tuple, Literal
from torch.nn import functional as F
from torch import nn

def dino_image_transform(image, patch_sz=14):
    if len(image.shape) == 4:
        b, c, h, w = image.shape
    elif len(image.shape) == 3:
        c, h, w = image.shape
        b = None
    elif len(image.shape) == 2:
        h, w = image.shape
        c = 1
        b = None
    h_new, w_new = (h // patch_sz) * patch_sz, (w // patch_sz) * patch_sz
    image_cropped = torchvision.transforms.CenterCrop((h_new, w_new))(image)
    return image_cropped

# Extract features from a Dino-v2 model
# this code is from https://github.com/AnyLoc/AnyLoc?tab=readme-ov-file#dinov2
_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", \
                        "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]
class DinoV2ExtractFeatures:
    """
        Extract features from an intermediate layer in Dino-v2
    """
    def __init__(self, dino_model: _DINO_V2_MODELS, layer: int, 
                facet: _DINO_FACETS="token", use_cls=False, 
                norm_descs=True, device: str = "cpu") -> None:
        """
            Parameters:
            - dino_model:   The DINO-v2 model to use
            - layer:        The layer to extract features from
            - facet:    "query", "key", or "value" for the attention
                        facets. "token" for the output of the layer.
            - use_cls:  If True, the CLS token (first item) is also
                        included in the returned list of descriptors.
                        Otherwise, only patch descriptors are used.
            - norm_descs:   If True, the descriptors are normalized
            - device:   PyTorch device to use
        """
        self.vit_type: str = dino_model
        self.dino_model: nn.Module = torch.hub.load(
                'facebookresearch/dinov2', dino_model)
        self.device = torch.device(device)
        self.dino_model = self.dino_model.eval().to(self.device)
        self.layer: int = layer
        self.facet = facet
        if self.facet == "token":
            self.fh_handle = self.dino_model.blocks[self.layer].\
                    register_forward_hook(
                            self._generate_forward_hook())
        else:
            self.fh_handle = self.dino_model.blocks[self.layer].\
                    attn.qkv.register_forward_hook(
                            self._generate_forward_hook())
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        # Hook data
        self._hook_out = None
    
    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output
        return _forward_hook
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
            Parameters:
            - img:   The input image
        """
        with torch.no_grad():
            res = self.dino_model(img)
            if self.use_cls:
                res = self._hook_out
            else:
                res = self._hook_out[:, 1:, ...]
            if self.facet in ["query", "key", "value"]:
                d_len = res.shape[2] // 3
                if self.facet == "query":
                    res = res[:, :, :d_len]
                elif self.facet == "key":
                    res = res[:, :, d_len:2*d_len]
                else:
                    res = res[:, :, 2*d_len:]
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None   # Reset the hook
        # returns [B, H*W, M]
        return res
    
    def __del__(self):
        self.fh_handle.remove()

class DinoFeatureExtractor:

    def __init__(self, version="dinov2_vits14", device="cuda"):
        self.dinov2_model = torch.hub.load('facebookresearch/dinov2', version)
        self.dinov2_model.to(device)
        self.dinov2_model.eval()  # Set to evaluation mode

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        # input images: [N, C, H, W] shape
        # returns features: [N, M, H, W]
        image_cropped = dino_image_transform(image)

        with torch.inference_mode():
            features = self.dinov2_model(image_cropped)

        #features = features.permute(0, 2, 3, 1)
        
        return features

def dino_pca_features(self, features, num_components=3, mask_bg=True, mask_bg_thresh=0.6):
    # features: [B, H, W, M] batch, height, width, embedding dimension
    pca = PCA(n_components=num_components)

    num_images, h, w, feat_size = features.shape
    
    pca_feats = pca.fit_transform(features.flatten(end_dim=-2).cpu().numpy())

    if mask_bg:
        pcaf_feats = minmax_scale(pca_feats[:,0:1]) > mask_bg_thresh
        pca_feats[~pcaf_feats.repeat(3,axis=1)] = np.min(pca_feats)
        
    pca_feats = (minmax_scale(pca_feats) * 255).astype('int')
    pca_feats = pca_feats.reshape(num_images, h, w, 3)
    # plot_image_grid(pca_feats, grid_size=(3,3), figsize=(6,6))

    return pca_feats


