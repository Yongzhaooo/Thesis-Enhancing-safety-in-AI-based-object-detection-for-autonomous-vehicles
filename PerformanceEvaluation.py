'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\Unsupervised_together_with_yolo_backbone\\sort\\PerformanceEvaluation.py
Author       : Yongzhao Chen
Date         : 2024-09-01 20:29:18
LastEditTime : 2024-09-01 22:38:06
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Describe & Note: 
This script evaluates the performance of a YOLO model on folders containing images and labels, with support for Out-of-Distribution (OOD) detection.
It performs the following steps:
1. Evaluates the mean Intersection over Union (IoU) for each class and across all classes in a folder.
2. Saves the evaluation results to a text file in the specified output folder.
3. Supports OOD detection and filtering of images and labels.
4. Copies valid images and labels with prefixes based on noise type and town information.
5. Outputs progress and completion status for each step.

The script is designed for evaluating folders of images and labels, with additional functionality for OOD detection and noise-based processing.
'''

import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from model_evaluation.Filter_OOD_image import filter_and_copy_images_with_ood_detection
from model_evaluation.NoiseFolderFinder import NoiseFolderFinder
from config.config_loader import load_config
from feature_processing.YoloBackboneAndPreprocess import YoloBackboneAndPreprocess
from model_evaluation.YOLOEvaluatorWithOOD import YOLOEvaluatorWithOOD
from model_evaluation.YOLO_Evaluator import YOLOEvaluator

def check_folder(img_folder, label_folder, output_folder, model_path, enable_single_evaluate=None):
    """Evaluate a folder of images and labels using a YOLO model."""
    # Create an instance of YOLOEvaluator
    evaluator = YOLOEvaluator(model_path)

    # Ensure the output folder exists
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Evaluate the entire folder
    mean_iou_per_class, mean_iou_across_all_classes = evaluator.evaluate_folder(img_folder, label_folder, enable_single_evaluate)

    # Save the evaluation results for the folder
    result_path = output_folder / 'folder_evaluation_results.txt'
    with open(result_path, 'w') as f:
        f.write('Mean IoU per class:\n')
        for cls, mean_iou in sorted(mean_iou_per_class.items()):
            f.write(f'Class {cls:.1f}: Mean IoU = {mean_iou:.5f}\n')
        f.write(f'Mean IoU across all classes: {mean_iou_across_all_classes:.5f}\n')

    print(f"Folder evaluation results saved to {result_path}")


class DataPreprocessorWithOOD:
    def __init__(self, folder_finder, ood_evaluator):
        """
        Initialize the class with NoiseFolderFinder and YOLOEvaluatorWithOOD instances.
        
        Args:
            folder_finder (NoiseFolderFinder): Used to find folders matching specific criteria.
            ood_evaluator: YOLOEvaluatorWithOOD instance for OOD detection.
        """
        self.folder_finder = folder_finder
        self.ood_evaluator = ood_evaluator
    
    def generate_prefix(self, folder):
        """
        Generate a prefix for files based on folder structure and noise type.
        
        Args:
            folder (str): Path to the folder.
        
        Returns:
            str: Generated prefix.
        """
        folder_name_parts = folder.split(os.sep)  # Split the path
        town_name = folder_name_parts[-4]  # Get the town name, e.g., Town07_Opt
        noise_type = folder_name_parts[-2]  # Get the noise type, e.g., gaussian or mosaic

        # Extract the town number, e.g., Town07 -> 07
        town_number = ''.join([char for char in town_name if char.isdigit()]).zfill(2)

        # Generate prefix based on noise type
        if noise_type == 'gaussian':
            # Extract sigma value, e.g., mean0_sigma1 -> 1
            sigma_value = folder_name_parts[-1].split('_')[-1].replace('sigma', '')
            prefix = f"T{town_number}_G{sigma_value}"
        elif noise_type == 'mosaic':
            # Extract num_mosaics value, e.g., num90_average -> 90
            num_mosaics = folder_name_parts[-1].split('_')[0].replace('num', '')
            prefix = f"T{town_number}_M{num_mosaics}"
        else:
            prefix = f"T{town_number}_raw"  # For raw data
        return prefix

    def copy_images_and_labels_with_prefix(self, valid_folders, test_folder_without_ood):
        """
        Copy images and labels from valid folders, adding prefixes to filenames.
        
        Args:
            valid_folders (List[str]): List of valid folder paths.
            test_folder_without_ood (str): Target folder for storing data without OOD.
        """
        test_folder_without_ood = Path(test_folder_without_ood)
        images_output_folder = test_folder_without_ood / 'images'
        labels_output_folder = test_folder_without_ood / 'labels'

        # Create target folders
        images_output_folder.mkdir(parents=True, exist_ok=True)
        labels_output_folder.mkdir(parents=True, exist_ok=True)

        for folder in valid_folders:
            # Generate prefix using the new generate_prefix function
            prefix = self.generate_prefix(folder)

            images_folder = os.path.join(folder, 'images')
            labels_folder = os.path.join(folder, 'labels')

            if os.path.exists(images_folder) and os.path.exists(labels_folder):
                # Copy images and labels, adding prefixes
                for image_file in os.listdir(images_folder):
                    if image_file.endswith(('.png', '.jpg')):
                        label_file = os.path.splitext(image_file)[0] + '.txt'
                        image_src = os.path.join(images_folder, image_file)
                        label_src = os.path.join(labels_folder, label_file)

                        if os.path.exists(label_src):
                            # New filenames
                            new_image_name = prefix + "_" + image_file
                            new_label_name = prefix + "_" + label_file

                            # Copy images
                            shutil.copy2(image_src, images_output_folder / new_image_name)

                            # Copy labels
                            shutil.copy2(label_src, labels_output_folder / new_label_name)

        print(f"All images and labels copied to {test_folder_without_ood} with prefixes added.")

    def apply_ood_detection(self, test_folder_without_ood, test_folder_with_ood):
        """
        Apply OOD detection to filter data and save results to a new folder.
        
        Args:
            test_folder_without_ood (str): Folder containing data without OOD.
            test_folder_with_ood (str): Target folder for filtered data with OOD detection applied.
        """
        test_folder_with_ood = Path(test_folder_with_ood)

        # Call the filter_and_copy_images_with_ood_detection function
        filter_and_copy_images_with_ood_detection(self.ood_evaluator, test_folder_without_ood, test_folder_with_ood)

        print(f"OOD filtering applied, results saved to {test_folder_with_ood}.")


# Example usage
if __name__ == "__main__":
    # Initialize NoiseFolderFinder and YOLOEvaluatorWithOOD instances
    top_level_layer = '/home/carla/yongzhao/thesis/finalevaluation/7towns/remix'
    folder_finder = NoiseFolderFinder(top_level_layer)

    # Find all valid folders
    folder_finder.find_all_csv()
    min_bias = 235
    max_bias = 245
    valid_folders = folder_finder.find_folders_by_bias_range(min_bias, max_bias)

    # Initialize YOLOEvaluator instance
    model_path = '/home/carla/yongzhao/thesis/thesis_yolo/best.pt'
    train_mean_vector_path = "/home/carla/yongzhao/thesis/datasets/20k_urban_train_vali/val/center/train_center.npy"
    config_path = "config/config.yaml"
    train_mean_vector = np.load(train_mean_vector_path)
    config = load_config(config_path)
    yolo_processor = YoloBackboneAndPreprocess(config)
    ood_threshold = 240
    evaluator = YOLOEvaluatorWithOOD(model_path, train_mean_vector, yolo_processor, ood_threshold=ood_threshold)

    # Create DataPreprocessorWithOOD instance
    preprocessor = DataPreprocessorWithOOD(folder_finder, evaluator)

    # Define folder for data without OOD
    test_folder_without_ood = '/home/carla/yongzhao/thesis/finalevaluation/test_folder_without_ood'

    # Copy valid images and labels with prefixes
    preprocessor.copy_images_and_labels_with_prefix(valid_folders, test_folder_without_ood)

    # Define folder for data with OOD filtering
    test_folder_with_ood = '/home/carla/yongzhao/thesis/finalevaluation/test_folder_with_ood'

    # Apply OOD detection and save results
    preprocessor.apply_ood_detection(test_folder_without_ood, test_folder_with_ood)
        
    # Evaluate folder without OOD
    print(f"Evaluating folder without OOD: {test_folder_without_ood}")
    check_folder(test_folder_without_ood + '/images', test_folder_without_ood + '/labels', test_folder_without_ood, model_path, enable_single_evaluate=None)
    
    # Evaluate folder with OOD
    print(f"Evaluating folder with OOD: {test_folder_with_ood}")
    check_folder(test_folder_with_ood + '/images', test_folder_with_ood + '/labels', test_folder_with_ood, model_path, enable_single_evaluate=None)
