"""
Description:
This script defines the `FullPipelineProcessor` class, which is responsible for processing image datasets 
with noise addition, evaluation, and result saving. The pipeline includes the following steps:
1. Loading configuration files and initializing necessary components.
2. Ensuring the existence of a training feature center file (`train_center.npy`).
3. Adding noise (e.g., Gaussian, mosaic) to images and evaluating the results.
4. Saving evaluation metrics such as bias and IoU to CSV files.
5. Cleaning up intermediate files and folders after processing.

The script is designed for batch processing of image datasets with various noise configurations.
"""

import os
import shutil
import numpy as np
import csv
from pathlib import Path
from feature_processing.YoloBackboneAndPreprocess import YoloBackboneAndPreprocess
from model_evaluation.YOLO_Evaluator import YOLOEvaluator
from config.config_loader import load_config
from data_processing.noise_processing import ImageNoiseAdder
from data_processing.NoiseConfigProcessor import NoiseConfigProcessor
import json
from data_processing.BiasCalculator import BiasCalculator
from itertools import chain

class FullPipelineProcessor:
    def __init__(self, input_folder_path, config_path):
        """Initialize the pipeline processor with input folder and configuration file."""
        # Load global configuration using the load_config function
        self.config = load_config(config_path)
        self.yolo_processor = YoloBackboneAndPreprocess(self.config)
        self.evaluator = YOLOEvaluator(self.config['model']['yolo_weights_path'])
        self.input_folder = input_folder_path
        os.makedirs(self.input_folder, exist_ok=True)
        self.output_folder = os.path.join(self.input_folder, 'with_noise')
        os.makedirs(self.output_folder, exist_ok=True)
        self.results_folder = os.path.join(self.input_folder, 'results')
        self.train_center_path = Path(self.config['paths']['train_data_path'] + '/center/train_center.npy')
        noise_paras_generator = NoiseConfigProcessor()
        noise_paras_generator.merge_noise_configs()
        
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

    def create_raw_folder_and_evaluate(self):
        """Create a raw folder, evaluate the original images, and copy labels."""
        print(f"Output folder: {self.output_folder}")
        
        # Ensure the raw folder path is correct
        raw_folder = os.path.join(self.output_folder, 'raw')
        os.makedirs(raw_folder, exist_ok=True)
        
        print(f"Raw folder: {raw_folder}")
        
        # Copy the images folder to the raw folder
        images_folder = os.path.join(self.input_folder, 'images')
        if os.path.exists(images_folder):
            shutil.copytree(images_folder, os.path.join(raw_folder, 'images'), dirs_exist_ok=True)
        else:
            print(f"Images folder not found: {images_folder}")
        
        # Copy labels
        self.copy_labels_to_folder(self.input_folder, raw_folder)

        # Evaluate the raw folder
        print("Evaluating the raw image folder...")
        self.evaluate_and_save_results(raw_folder, 'raw')

    def copy_labels_to_folder(self, source_folder, destination_folder):
        """Copy labels from the source folder to the destination folder."""
        labels_folder = os.path.join(source_folder, 'labels')
        if os.path.exists(labels_folder):
            shutil.copytree(labels_folder, os.path.join(destination_folder, 'labels'), dirs_exist_ok=True)
        else:
            print(f"Labels folder not found: {labels_folder}")
            
    def single_folder_with_certain_noise(self, noise_config_path):
        """Load noise parameter file, generate noise folders, evaluate, and clean up."""
        # Load noise parameters
        with open(noise_config_path, 'r') as f:
            noise_params = json.load(f)
        
        # Process Gaussian noise
        for gaussian_param in noise_params['gaussian']:
            noise_adder = ImageNoiseAdder(self.input_folder, self.output_folder, self.config)
            noise_adder.apply_noise_to_images(noise_type="gaussian", **gaussian_param)
            self.process_single_noise_folder(noise_adder.output_folder, "gaussian", gaussian_param)

        # Process mosaic noise
        for mosaic_param in noise_params['mosaic']:
            noise_adder = ImageNoiseAdder(self.input_folder, self.output_folder, self.config)
            noise_adder.apply_noise_to_images(noise_type="mosaic", **mosaic_param)
            self.process_single_noise_folder(noise_adder.output_folder, "mosaic", mosaic_param)

    def process_single_noise_folder(self, folder_path, noise_type, noise_params):
        """Evaluate a single noise folder, save results, and clean up."""
        folder_path = Path(folder_path)
        print(f"Running evaluation for {noise_type} noise in {folder_path}...")

        img_folder = folder_path / 'images'
        label_folder = folder_path / 'labels'
        if not (img_folder.exists() and label_folder.exists()):
            print(f"Images or labels are missing in {folder_path}")
            return  # Exit early if images or labels are missing

        # Convert noise parameter dictionary to descriptive string
        noise_param_str = f"{noise_type}_params: " + ", ".join([f"{k}={v}" for k, v in noise_params.items()])
        self.evaluate_and_save_results(folder_path, noise_type, noise_param_str)

    def evaluate_and_save_results(self, folder_path, noise_type, noise_params=None):
        """Evaluate the folder and save results."""
        img_folder = Path(folder_path) / 'images'
        label_folder = Path(folder_path) / 'labels'
    
        # Use BiasCalculator to calculate bias and save histinfo
        bias_calculator = BiasCalculator(self.train_center_path, folder_path, self.yolo_processor)
        
        # 1. Calculate bias for the entire folder and save as bias_results.json
        avg_bias, bias_list = bias_calculator.calculate_bias_for_folder(output_json=f"{folder_path}/bias_results.json")
        
        # 2. Compute and save histinfo to the folder
        bias_calculator.compute_histinfo(bias_list, bin_size=2)
        bias_calculator.save_histinfo(output_json=f"{folder_path}/histinfo.json")
    
        # 3. Calculate and save IoU results
        mean_iou_per_class, mean_iou_across_all_classes = self.evaluator.evaluate_folder(img_folder, label_folder)
    
        # 4. Save bias and IoU results to CSV
        self.save_results_to_csv(noise_type, avg_bias, mean_iou_across_all_classes, folder_path, noise_params)
    
        # Clean up the contents of img_folder and label_folder
        if img_folder.exists():
            shutil.rmtree(img_folder)
        if label_folder.exists():
            shutil.rmtree(label_folder)

    def save_results_to_csv(self, noise_type, bias, mean_iou_across_all_classes, folder_path, noise_param_str=None):
        """Save results to a general CSV and a noise-specific CSV file."""
        folder_name = Path(folder_path).name
        results_folder = Path(self.results_folder) / noise_type
        results_folder.mkdir(parents=True, exist_ok=True)

        # Path for the general results CSV
        general_result_file = Path(self.results_folder) / 'overall_evaluation_results.csv'

        # Path for the noise-specific results CSV
        noise_specific_file = results_folder / f'{noise_type}_{folder_name}_results.csv'

        # Convert bias and mean IoU to float
        bias = float(bias)
        mean_iou_across_all_classes = float(mean_iou_across_all_classes)

        # Save to the noise-specific CSV file
        with open(noise_specific_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Noise_Type_Params', 'Bias', 'Mean IoU Across All Classes'])
            writer.writerow([noise_param_str, bias, mean_iou_across_all_classes])

        # Append to the general overall CSV file
        write_header = not general_result_file.exists()  # Check if the header needs to be written
        with open(general_result_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(['Noise_Type_Params', 'Bias', 'Mean IoU Across All Classes'])
            writer.writerow([noise_param_str, bias, mean_iou_across_all_classes])

    def run(self):
        """Run the full pipeline process."""
        self.ensure_train_center_exists()
        self.create_raw_folder_and_evaluate()
        # Iterate through noise parameter files and process each noise folder
        noise_config_path = os.path.join('config/noise_configs', 'noise_params.json')
        print(f"Processing noise parameter file: {noise_config_path}")
        self.single_folder_with_certain_noise(noise_config_path)

if __name__ == "__main__":
    # Load configuration file using config_loader
    input_folder_path = '/home/carla/yongzhao/thesis/finalevaluation/7towns/30pics/Town01_Opt/'
    config_path = 'config/config.yaml'
    processor = FullPipelineProcessor(input_folder_path, config_path)
    processor.run()
