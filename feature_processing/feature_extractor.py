'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\Unsupervised_together_with_yolo_backbone\\sort\\feature_processing\\feature_extractor.py
Author       : Yongzhao Chen
Date         : 2024-09-01 15:13:44
LastEditTime : 2024-09-01 15:42:18
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Description:
This script defines two feature extractor classes: `YoloFeatureExtractor` and `ResNetFeatureExtractor`.
It performs the following tasks:
1. Extracts features from images using YOLO or ResNet models.
2. Supports batch processing for efficient feature extraction.
3. Handles both training and inference modes, with configurable paths and batch sizes.
4. Applies preprocessing steps such as resizing, normalization, and tensor conversion.

The script is designed for feature extraction in machine learning pipelines, enabling downstream tasks such as classification or clustering.
'''

import os
import numpy as np
import torch
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
from feature_processing.YoloBackboneAndPreprocess import YoloBackboneAndPreprocess

class YoloFeatureExtractor:
    def __init__(self, config, image_paths, infer_mode=False):
        """
        Initialize the YOLO feature extractor.

        Args:
            config (dict): Configuration dictionary containing paths and settings.
            image_paths (list): List of image file names to process.
            infer_mode (bool): Whether to run in inference mode. Defaults to False.
        """
        self.config = config
        self.image_paths = image_paths
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.infer_mode = infer_mode
        self.yolo_processor = YoloBackboneAndPreprocess(config, self.infer_mode)

    def extract_features(self):
        """
        Extract features from images using the YOLO model.

        Returns:
            np.ndarray: Array of extracted features.
        """
        features = []
        base_path = os.path.join(self.config['infer_data_path'], 'images') if self.infer_mode else os.path.join(self.config['train_data_path'], 'images')

        full_image_paths = [os.path.join(base_path, path) for path in self.image_paths]

        for i in tqdm(range(0, len(full_image_paths), self.config['batch_size']), desc="Extracting YOLO features"):
            batch_paths = full_image_paths[i:i + self.config['batch_size']]
            batch_features = self.yolo_processor.calculate_features_for_batch(batch_paths)
            if batch_features is not None:
                features.extend(batch_features.cpu().numpy().reshape(batch_features.shape[0], -1))
            else:
                print(f"Failed to extract features for batch starting at index {i}")

        return np.array(features)


class ResNetFeatureExtractor:
    def __init__(self, config, image_paths, infer_mode=False):
        """
        Initialize the ResNet feature extractor.

        Args:
            config (dict): Configuration dictionary containing paths and settings.
            image_paths (list): List of image file names to process.
            infer_mode (bool): Whether to run in inference mode. Defaults to False.
        """
        self.config = config
        self.image_paths = image_paths
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=True).to(self.device)
        self.model.eval()
        if self.config['freeze_weights']:
            for param in self.model.parameters():
                param.requires_grad = False

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.infer_mode = infer_mode

    def extract_features(self):
        """
        Extract features from images using the ResNet model.

        Returns:
            np.ndarray: Array of extracted features.
        """
        features = []
        base_path = os.path.join(self.config['infer_data_path'], 'images') if self.infer_mode else os.path.join(self.config['train_data_path'], 'images')

        for i in tqdm(range(0, len(self.image_paths), self.config['batch_size']), desc="Extracting ResNet features"):
            batch_paths = [os.path.join(base_path, image_path) for image_path in self.image_paths[i:i + self.config['batch_size']]]
            batch_images = [self.transform(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device) for path in batch_paths]
            batch_images = torch.cat(batch_images, dim=0)

            with torch.no_grad():
                batch_features = self.model(batch_images)
                features.extend(batch_features.cpu().numpy())
        return np.array(features)
