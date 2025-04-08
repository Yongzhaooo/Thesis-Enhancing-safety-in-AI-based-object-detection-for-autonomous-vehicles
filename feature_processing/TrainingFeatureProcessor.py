'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\Unsupervised_together_with_yolo_backbone\\sort\\feature_processing\\TrainingFeatureProcessor.py
Author       : Yongzhao Chen
Date         : 2024-09-01 15:13:44
LastEditTime : 2024-09-01 15:42:18
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Description:
This script defines the `TrainingFeatureProcessor` class, which is responsible for processing training data 
to extract features and calculate the feature center. It performs the following tasks:
1. Loads configuration settings from a YAML file.
2. Extracts features from training images using the YOLO model.
3. Saves the extracted features to a file.
4. Calculates the feature center from the extracted features.
5. Saves the calculated feature center to a file.
6. Handles existing feature and center files to avoid redundant processing.

The script is designed for preparing training data in machine learning pipelines, enabling downstream tasks such as clustering or classification.
'''

import os
import numpy as np
import torch
from tqdm import tqdm
from feature_processing.YoloBackboneAndPreprocess import YoloBackboneAndPreprocess
import yaml

def load_config(config_path='config/config.yaml'):
    """
    Load the global configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

class TrainingFeatureProcessor:
    def __init__(self, config):
        """
        Initialize the TrainingFeatureProcessor with the given configuration.

        Args:
            config (dict): Configuration dictionary containing model and path settings.
        """
        self.config = config['model']  # Model-related configurations
        self.paths = config['paths']  # Path-related configurations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_processor = YoloBackboneAndPreprocess(config)
        self.train_data_path = self.paths['train_data_path']
        self.images_path = os.path.join(self.train_data_path, 'images')
        self.feature_save_path = os.path.join(self.train_data_path, 'feature.npy')
        self.center_save_path = os.path.join(self.train_data_path, 'center', 'train_center.npy')

    def extract_features(self, image_paths):
        """
        Extract features from the given list of image paths.

        Args:
            image_paths (list): List of image file names.

        Returns:
            np.ndarray: Extracted features as a NumPy array.
        """
        features = []
        valid_image_paths = [os.path.join(self.images_path, path) for path in image_paths if os.path.exists(os.path.join(self.images_path, path))]
        
        if len(valid_image_paths) != len(image_paths):
            print(f"Warning: {len(image_paths) - len(valid_image_paths)} images not found and will be skipped.")

        for i in tqdm(range(0, len(valid_image_paths), self.config['batch_size']), desc="Extracting YOLO features"):
            batch_paths = valid_image_paths[i:i + self.config['batch_size']]
            batch_features = self.yolo_processor.calculate_features_for_batch(batch_paths)
            if batch_features is not None:
                batch_features = batch_features.cpu().numpy()
                batch_features = batch_features.reshape(batch_features.shape[0], -1)
                features.extend(batch_features)
            else:
                print(f"Failed to extract features for batch starting at index {i}")

        return np.array(features)

    def save_features(self, features):
        """
        Save the extracted features to a file.

        Args:
            features (np.ndarray): Extracted features.
        """
        os.makedirs(os.path.dirname(self.feature_save_path), exist_ok=True)
        np.save(self.feature_save_path, features)
        print(f"Features saved to {self.feature_save_path}")

    def calculate_center(self, features):
        """
        Calculate the feature center from the extracted features.

        Args:
            features (np.ndarray): Extracted features.

        Returns:
            np.ndarray: Calculated feature center.
        """
        if features.size == 0:
            raise ValueError("Features array is empty. Cannot calculate center.")
        return np.mean(features, axis=0)

    def save_center(self, center):
        """
        Save the calculated feature center to a file.

        Args:
            center (np.ndarray): Calculated feature center.
        """
        os.makedirs(os.path.dirname(self.center_save_path), exist_ok=True)
        np.save(self.center_save_path, center)
        print(f"Center saved to {self.center_save_path}")

    def process_training_data(self):
        """
        Process the training data to extract features and calculate the feature center.

        Returns:
            tuple: Extracted features and calculated feature center.
        """
        # Check if feature and center files already exist
        if os.path.exists(self.feature_save_path) and os.path.exists(self.center_save_path):
            print(f"Features and center already exist. Skipping processing.")
            # Load and return existing features and center
            train_features = np.load(self.feature_save_path)
            train_center = np.load(self.center_save_path)
            return train_features, train_center

        # Get the list of training image paths
        if not os.path.exists(self.images_path):
            raise FileNotFoundError(f"Images path {self.images_path} does not exist.")
        
        image_paths = os.listdir(self.images_path)
        if not image_paths:
            raise ValueError("No images found in images path.")
        
        # Extract features from training images
        train_features = self.extract_features(image_paths)
        if train_features.size == 0:
            raise ValueError("No features extracted from training data.")
        
        # Save the extracted features
        self.save_features(train_features)
        
        # Calculate the feature center
        train_center = self.calculate_center(train_features)
        
        # Save the calculated feature center
        self.save_center(train_center)
        
        print("Training data processing complete.")
        return train_features, train_center  # Return features and center

if __name__ == "__main__":
    config = load_config()
    processor = TrainingFeatureProcessor(config)
    features, center = processor.process_training_data()
    
    # Output results for verification
    print(f"Extracted features shape: {features.shape}")
    print(f"Calculated center shape: {center.shape}")
