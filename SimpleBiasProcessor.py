'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\Unsupervised_together_with_yolo_backbone\\sort\\SimpleBiasProcessor.py
Author       : Yongzhao Chen
Date         : 2024-09-01 20:29:18
LastEditTime : 2024-09-01 22:38:06
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Describe & Note: 
This script defines the `SimpleBiasProcessor` class, which processes image datasets to calculate bias values.
It performs the following steps:
1. Ensures the existence of a training feature center file (`train_center.npy`).
2. Calculates the feature center for training data if the file does not exist.
3. Processes each folder containing `images` and `labels` subdirectories to compute bias values.
4. Saves bias results and histogram information for each folder.
5. Appends the results to a summary CSV file.

The script is designed for batch processing of multiple subfolders in a directory.
'''

import os
import numpy as np
import csv
from pathlib import Path
from feature_processing.YoloBackboneAndPreprocess import YoloBackboneAndPreprocess
from config.config_loader import load_config
from itertools import chain
from data_processing.BiasCalculator import BiasCalculator

class SimpleBiasProcessor:
    def __init__(self, top_level_folder, config_path):
        """Initialize the processor with the top-level folder and configuration file."""
        # Load global configuration using the load_config function
        self.config = load_config(config_path)
        self.yolo_processor = YoloBackboneAndPreprocess(self.config)
        self.top_level_folder = top_level_folder
        self.results_folder = Path(self.top_level_folder) / 'results'
        os.makedirs(self.results_folder, exist_ok=True)
        self.train_center_path = Path(self.config['paths']['train_data_path'] + '/center/train_center.npy')

    def ensure_train_center_exists(self):
        """Ensure the train_center.npy file exists. If not, calculate and save it."""
        if not self.train_center_path.exists():
            print(f"Train center file not found at {self.train_center_path}. Calculating...")
            train_center = self.calculate_feature_center(self.config['paths']['train_data_path'])
            self.save_train_center(train_center)
        else:
            print(f"Train center file found: {self.train_center_path}")

    def calculate_feature_center(self, image_folder):
        """Calculate the feature center of training data and return it."""
        print(f"Calculating feature center for images in: {image_folder}")
        all_features = []
        image_folder = Path(image_folder) / 'images'

        # Iterate through each image in the folder and calculate feature vectors
        image_paths = chain(image_folder.glob('*.png'), image_folder.glob('*.jpg'))
        for image_path in image_paths:
            feature_vector = self.yolo_processor.calculate_feature_vector(image_path).cpu().numpy().flatten()
            all_features.append(feature_vector)

        # Return the mean of feature vectors as the training center
        return np.mean(all_features, axis=0)

    def save_train_center(self, train_center):
        """Save the training center to the train_center.npy file."""
        train_center_dir = self.train_center_path.parent
        train_center_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        np.save(self.train_center_path, train_center)
        print(f"Train center saved to {self.train_center_path}")

    def process_folder(self, folder_path):
        """Process a single folder, calculate bias, and save results."""
        folder_path = Path(folder_path)
        print(f"Processing folder: {folder_path}")

        # Use BiasCalculator to calculate bias and save histogram information
        bias_calculator = BiasCalculator(self.train_center_path, folder_path, self.yolo_processor)

        # Calculate bias for the entire folder and save as bias_results.json
        avg_bias, bias_list = bias_calculator.calculate_bias_for_folder(output_json=f"{folder_path}/bias_results.json")

        # Compute and save histogram information to the folder
        bias_calculator.compute_histinfo(bias_list, bin_size=2)
        bias_calculator.save_histinfo(output_json=f"{folder_path}/histinfo.json")

        # Save bias results, setting IoU to 0
        self.save_results_to_csv(folder_path, avg_bias, iou=0)

    def save_results_to_csv(self, folder_path, avg_bias, iou):
        """Save results to the summary results CSV file."""
        folder_name = Path(folder_path).name

        # Path for the general results CSV
        general_result_file = self.results_folder / 'overall_evaluation_results.csv'

        # Check if the header needs to be written
        write_header = not general_result_file.exists()

        # Append to the general overall CSV file
        with open(general_result_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(['Noise_Type_Params', 'Bias', 'Mean IoU Across All Classes'])
            writer.writerow([f"{folder_name}_raw", avg_bias, iou])

    def run(self):
        """Iterate through all subfolders containing `images` and `labels` and process each folder."""
        self.ensure_train_center_exists()

        for root, dirs, files in os.walk(self.top_level_folder):
            if 'images' in dirs and 'labels' in dirs:
                self.process_folder(root)

if __name__ == "__main__":
    # Define the top-level folder path
    top_level_folder = '/home/carla/yongzhao/thesis/finalevaluation/new_coco_train_500/'
    config_path = 'config/config.yaml'

    # Create an instance of SimpleBiasProcessor and run it
    processor = SimpleBiasProcessor(top_level_folder, config_path)
    processor.run()
