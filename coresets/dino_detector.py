import cv2
import sys

import torch
import numpy as np
from torchvision import transforms
import argparse
import time

import clustering.knn_mask
import feature_extraction.dino_utils
# import viz_utils

def torch2cv_img(torch_img):
    torch_img = (torch_img * 255).byte().cpu().numpy()  # Convert to numpy array and scale to [0, 255]
    return torch_img

def torch2cv_shape(torch_img):
    """
    Convert a tensor from (B, C, H, W) to (H, W, C) format for OpenCV compatibility.
    """
    torch_img = torch_img.squeeze(0)  # Remove batch dimension
    torch_img = torch_img.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
    return torch_img

class FeatureExtractor:
    def __init__(self, model_name='featup/dinov2', device='cuda'):
        self.device = device
        self.model_name = model_name
        self.patch_size = None

        # Load the pre-trained DINO model
        if model_name == "featup/dinov2":
            self.model = torch.hub.load("mhamilton723/FeatUp", 'dinov2').eval().to(device)
            self.input_patch_size = 14
            self.output_patch_size = 1

        # Todo: support further models and facets
        elif model_name == "anyloc/dinov2_vits14/token":
            self.model = feature_extraction.dino_utils.DinoV2ExtractFeatures("dinov2_vits14", -1, "token", device=device)
            self.input_patch_size = 14
            self.output_patch_size = 14

        else:
            raise ValueError(f"Unsupported model name: {model_name}")

    def __call__(self, *args, **kwds):
        return self.extract(*args, **kwds)
    
    def preprocess(self, frame):
        # Preprocess the frame (DINO-formatting)
        frame = transforms.ToTensor()(frame).unsqueeze(0)  # Convert to tensor and add batch dimension
        frame_cropped = feature_extraction.dino_utils.dino_image_transform(frame, patch_sz=self.input_patch_size).to(self.device)
        return frame_cropped

    def extract(self, frame):
        """
        Input: frame (H_in, W_in, C) in BGR format
        Output: features (H_out, W_out, M) where M is the feature dimension
        """
        # Preprocess the frame to torch format: [B, C, H, W]
        frame_cropped = self.preprocess(frame)
        
        # Extract features
        with torch.no_grad():
            features = self.model(frame_cropped)

        # Post-process features
        if "anyloc/dinov2_vits" in self.model_name:
            batch_size, n_channels, h_patch, w_patch = frame_cropped.shape
            h_feats = h_patch // self.output_patch_size
            w_feats = w_patch // self.output_patch_size
            features = features.reshape(batch_size, h_feats, w_feats, -1)
            features = features.permute(0, 3, 1, 2)

        features = torch2cv_shape(features)  # Convert to (H, W, M) format

        # Reshape to cv2 style output
        frame_cropped = torch2cv_shape(frame_cropped)

        return features, frame_cropped

class StreamingDetector:
    def __init__(self, feature_extractor: FeatureExtractor, 
                 target_features=[], 
                 target_labels=[],
                 dist_func = "cosine_similarity",
                 resolution = (640, 480),
                 colors = None,
                 device='cuda'):
        
        self.device = device
        self.feature_extractor = feature_extractor
        self.dist_func = dist_func

        self.target_features = target_features
        self.target_labels = target_labels

        self.curr_frame = None # H, W, C
        self.curr_frame_cropped = None # B, C, H, W format used as input to DINO
        self.curr_features = None # H_feat, W_feat, M
        self.curr_labels = [] # H_feat, W_feat
        self.curr_dists = [] # H_feat, W_feat

        self.curr_label_mode = 0

        self.resolution = resolution  # (width, height)

        if colors is None:
            self.colors = [
                (255, 0, 0),    # Blue
                (0, 255, 0),    # Green
                (0, 0, 255),    # Red
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (0, 0, 0),      # Black
                (255, 255, 255) # White
            ]
        else:
            self.colors = colors

        self.clicked_points = []
        self.clicked_labels = []

    def detect(self, frame):
        # Extract features
        features, frame_cropped = self.feature_extractor(frame)

        # Find classes
        labels, dists = clustering.knn_mask.nn_classify(features, 
                                                           self.target_features, 
                                                           self.target_labels, 
                                                           dist_func=self.dist_func)
        return labels, dists, features, frame_cropped

    def add_target_features(self, features, label):
        self.target_features.append(features)
        self.target_labels.append(label)

    def query_features(self, x, y):
        """
        Maps image coordinates (x, y) to feature coordinates and returns the feature vector at that location
        """
        patch_size = self.feature_extractor.output_patch_size
        qx = x // patch_size
        qy = y // patch_size

        if self.curr_features is not None:
            return self.curr_features[qy, qx, :]
        else:
            return None
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_target_features(self.query_features(x, y), self.curr_label_mode)
            self.clicked_points.append((x, y))
            self.clicked_labels.append(self.curr_label_mode)

    def run(self, vidpath):
        # Initialize the playback window with first frame
        cap = cv2.VideoCapture(vidpath)
        ret, self.curr_frame = cap.read()
        
        self.curr_frame = cv2.resize(self.curr_frame, self.resolution)  # Resize to resolution

        self.curr_features, self.curr_frame_cropped = self.feature_extractor.extract(self.curr_frame)

        cv2.namedWindow("Frame")
        cv2.setMouseCallback("Frame", self.mouse_callback)
        cv2.imshow('Frame', self.curr_frame)
        paused = True

        # Playback
        while cap.isOpened():
                
            if not paused:
                ret, self.curr_frame = cap.read()
                self.curr_frame = cv2.resize(self.curr_frame, self.resolution)

                self.curr_features, self.curr_frame_cropped = self.feature_extractor.extract(self.curr_frame)
                if not ret:
                    break

            if len(self.target_features) > 0:
                # Get labels
                self.curr_labels, self.curr_dists, self.curr_features, _ = self.detect(self.curr_frame)

                # Display the current frame with labels
                h_feats, w_feats = self.curr_labels.shape
                label_image = np.zeros((h_feats, w_feats, 3), dtype=np.uint8)
                for i, color in enumerate(self.colors):
                    label_inds = (self.curr_labels == i).cpu().numpy()
                    label_image[label_inds] = color

                height, width = self.curr_frame_cropped.shape[:2]
                label_image = label_image * (1 - self.curr_dists.cpu().numpy()[:,:,np.newaxis]) # This only works with cosine similarity at the moment
                label_image = cv2.resize(label_image.astype(np.uint8), (width, height), interpolation=cv2.INTER_CUBIC)
                
                combined_image = cv2.addWeighted(torch2cv_img(self.curr_frame_cropped), 0.7, label_image, 0.3, 0)

                # print("labels shape: ", self.curr_labels.shape)
                # print("dists shape: ", self.curr_dists.shape)
                # print("features shape: ", self.curr_features.shape)
                # print("dists min/max: ", self.curr_dists.min(), self.curr_dists.max())
            else:
                combined_image = torch2cv_img(self.curr_frame_cropped)

            for i, click in enumerate(self.clicked_points):
                cv2.circle(combined_image, (click[0], click[1]), 5, self.colors[self.clicked_labels[i]], -1)

            cv2.imshow('Frame', combined_image)

            key = cv2.waitKey(1)

            for i in range(len(self.colors)):
                if key==ord(str(i)):
                    self.curr_label_mode = i
                    print("Class mode: ", self.curr_label_mode)

            if key == ord(' '):
                paused = not paused

            # Exit if the user presses 'q'
            if key == ord('q'):
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple masking/detector-ish program to test feature extractors"
    )
    parser.add_argument("vidpath", type=str)
    parser.add_argument("--model_name", 
                        type=str, 
                        default="anyloc/dinov2_vits14/token", 
                        help="One of featup/dinov2 or anyloc/dinov2_vits14/token")


    args = parser.parse_args()

    feat_extractor = FeatureExtractor(model_name=args.model_name, device='cuda')
    detector = StreamingDetector(feat_extractor)
    detector.run(args.vidpath)