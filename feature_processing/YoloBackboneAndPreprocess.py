'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\Unsupervised_together_with_yolo_backbone\\sort\\feature_processing\\YoloBackboneAndPreprocess.py
Author       : Yongzhao Chen
Date         : 2024-09-01 15:13:44
LastEditTime : 2024-09-01 15:42:18
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Description:
This script defines the `YoloBackboneAndPreprocess` class and the `LoadImagesOnly` dataset class. It performs the following tasks:
1. Loads YOLO model and extracts features from images using its backbone.
2. Supports batch processing for efficient feature extraction.
3. Handles preprocessing steps such as resizing, normalization, and tensor conversion.
4. Assigns pretrained weights to the YOLO backbone for feature extraction.
5. Provides utility functions for loading and processing images in batches.

The script is designed for feature extraction in machine learning pipelines, enabling downstream tasks such as clustering or classification.
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
import glob
import sys
import os

# Import from your custom yolotool module
from feature_processing.yolotool import LoadImages, letterbox

# Add YOLO model path to the sys.path
sys.path.append('/home/carla/yongzhao/thesis/thesis_yolo/')
from models.yolo import Model
import pathlib

# Ensure pathlib handles different OS paths correctly
pathlib.WindowsPath = pathlib.PosixPath

class LoadImagesOnly(Dataset):
    def __init__(self, path, img_size=640, batch_size=32, rect=False, stride=32, pad=0.0):
        """
        Dataset class for loading images from a directory or file list.

        Args:
            path (str or list): Path to the directory or list of image paths.
            img_size (int): Target image size for resizing.
            batch_size (int): Number of images per batch.
            rect (bool): Whether to use rectangular resizing. Defaults to False.
            stride (int): Stride for resizing. Defaults to 32.
            pad (float): Padding value. Defaults to 0.0.
        """
        self.img_size = img_size
        self.stride = stride
        self.rect = rect
        self.batch_size = batch_size

        # Load files
        files = []
        for p in path if isinstance(path, list) else [path]:
            p = Path(p)
            if p.is_dir():
                files.extend(sorted(glob.glob(str(p / '**' / '*.*'), recursive=True)))
            elif p.is_file():
                with open(p) as f:
                    files.extend(f.read().strip().splitlines())
            else:
                raise FileNotFoundError(f"{p} does not exist")
        self.files = sorted([x for x in files if x.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']])
        
        if not self.files:
            raise FileNotFoundError(f"No image files found in {path}")

        print(f"Found {len(self.files)} image files in {path}")

        # Calculate batches
        self.batch = np.floor(np.arange(len(self.files)) / self.batch_size).astype(int)
        self.nb = self.batch[-1] + 1  # number of batches

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """
        Load and preprocess a single image.

        Args:
            index (int): Index of the image to load.

        Returns:
            np.ndarray: Preprocessed image.
        """
        path = self.files[index]
        img = cv2.imread(path)  # BGR
        if img is None:
            print(f"Failed to load image {path}")
            return None
        img = letterbox(img, new_shape=self.img_size, stride=self.stride, auto=self.rect)[0]
        img = img.transpose(2, 0, 1)  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return img
    
    def get_batch(self, batch_index):
        """
        Get a batch of images by batch index.

        Args:
            batch_index (int): Index of the batch.

        Returns:
            np.ndarray: Batch of images.
        """
        indices = np.where(self.batch == batch_index)[0]
        images = [self.__getitem__(i) for i in indices]
        images = [img for img in images if img is not None]  # Remove any None values due to failed loads
        if not images:
            print(f"No valid images found for batch index {batch_index}")
            return None
        return np.stack(images, axis=0)  # stack images to form a batch
    
    def get_batch_by_paths(self, paths):
        """
        Get a batch of images by their file paths.

        Args:
            paths (list): List of image file paths.

        Returns:
            np.ndarray: Batch of images.
        """
        images = [self.__getitem__(self.files.index(path)) for path in paths if path in self.files]
        images = [img for img in images if img is not None]  # Remove any None values due to failed loads
        if not images:
            print("No valid images found in provided paths.")
            return None
        return np.stack(images, axis=0)  # stack images to form a batch
    
class YoloBackboneAndPreprocess:
    def __init__(self, config, infer_mode=False):
        """
        Initialize the YOLO backbone and preprocessing pipeline.

        Args:
            config (dict): Configuration dictionary containing model and path settings.
            infer_mode (bool): Whether to run in inference mode. Defaults to False.
        """
        self.config = config
        model_config = config['model']
        self.device = torch.device(model_config.get('device', "cuda" if torch.cuda.is_available() else "cpu"))
        self.img_size = model_config['img_size']  # Image size from the model configuration
        self.model = Model(cfg=model_config['yolo_config_path']).to(self.device)
        self.backbone = nn.Sequential(*list(self.model.model.children())[:10])
        self.infer_mode = infer_mode

        # Load weights
        self.assign_weights()

        # Create data loader
        if not self.infer_mode:
            self.loader = LoadImagesOnly(config['paths']['train_data_path'], img_size=self.img_size)
        else:
            inferimages = os.path.join(config['paths']['infer_data_path'], 'images')
            self.loader = LoadImagesOnly(inferimages, img_size=self.img_size)

    def process_as_yolo_do(self, image_path):
        """
        Preprocess an image as YOLO does.

        Args:
            image_path (str): Path to the image.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        dataset = LoadImages(image_path, img_size=self.img_size, stride=32, auto=True)
        for path, img, im0s, vid_cap, _ in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.float() / 255.0
            img = img.unsqueeze(0)
            return img

    def calculate_feature_vector(self, image_path):
        """
        Calculate the feature vector for a single image.

        Args:
            image_path (str): Path to the image.

        Returns:
            torch.Tensor: Feature vector.
        """
        processed_img = self.process_as_yolo_do(image_path)
        with torch.no_grad():
            features = self.backbone(processed_img)
        return features

    def calculate_features_for_batch(self, image_paths):
        """
        Calculate feature vectors for a batch of images.

        Args:
            image_paths (list): List of image file paths.

        Returns:
            torch.Tensor: Batch of feature vectors.
        """
        images = self.loader.get_batch_by_paths(image_paths)
        if images is None:
            print("Failed to load images.")
            return None

        images = torch.from_numpy(images).to(self.device).float() / 255.0
        
        with torch.no_grad():
            features = self.backbone(images)
        return features

    def assign_weights(self):
        """
        Assign pretrained weights to the YOLO backbone.
        """
        pretrained_weights_path = self.config['model']['yolo_weights_path']
        try:
            pretrained_dict = torch.load(pretrained_weights_path, map_location=self.device)["model"].float().state_dict()
        except FileNotFoundError:
            print(f"Pretrained weights not found at {pretrained_weights_path}")
            return

        if not pretrained_dict:
            print("No pretrained weights found in the loaded file.")
            return

        print("Pretrained weights loaded successfully.")
        
        backbone_dict = self.backbone.state_dict()
        for name in backbone_dict.keys():
            pretrained_key = f'model.{name}'
            if pretrained_key in pretrained_dict:
                backbone_dict[name] = pretrained_dict[pretrained_key]

        self.backbone.load_state_dict(backbone_dict)
        print("Weights assigned successfully.")
