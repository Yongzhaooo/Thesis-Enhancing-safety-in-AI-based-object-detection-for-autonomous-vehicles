'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\sort\\test.py
Author       : Yongzhao Chen
Date         : 2024-09-17 14:58:15
LastEditTime : 2024-09-17 14:59:45
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Describe & Note: 
'''
import numpy as np
from pathlib import Path
from model_evaluation.YOLOEvaluatorWithOOD import YOLOEvaluatorWithOOD
from feature_processing.YoloBackboneAndPreprocess import YoloBackboneAndPreprocess
from config.config_loader import load_config
import shutil

def filter_and_copy_images_with_ood_detection(evaluator, input_folder, output_folder):
    """
    Filters images based on OOD detection and copies non-OOD images and their labels to the output folder.

    Args:
        evaluator: An instance of YOLOEvaluatorWithOOD for OOD detection.
        input_folder: Path to the input folder containing 'images' and 'labels' subfolders.
        output_folder: Path to the output folder where filtered images and labels will be saved.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    images_input_folder = input_folder / 'images'
    labels_input_folder = input_folder / 'labels'
    images_output_folder = output_folder / 'images'
    labels_output_folder = output_folder / 'labels'

    # Create output directories if they don't exist
    images_output_folder.mkdir(parents=True, exist_ok=True)
    labels_output_folder.mkdir(parents=True, exist_ok=True)

    img_files = list(images_input_folder.glob('*.png')) + list(images_input_folder.glob('*.jpg'))

    for img_file in img_files:
        label_file = labels_input_folder / (img_file.stem + '.txt')
        if label_file.exists():
            # Perform OOD detection
            is_in_distribution = evaluator.run_ood_detection(img_file)
            if is_in_distribution:
                # Copy image and label to output folder
                shutil.copy2(img_file, images_output_folder / img_file.name)
                shutil.copy2(label_file, labels_output_folder / label_file.name)
            else:
                # Image is OOD, do not copy
                pass
        else:
            print(f"Label file not found for image {img_file}")



if __name__ == "__main__":
    # Paths and parameters
    model_path = '/home/carla/yongzhao/thesis/thesis_yolo/best.pt'
    train_mean_vector_path = "/home/carla/yongzhao/thesis/datasets/20k_urban_train_vali/val/center/train_center.npy"
    config_path = "config/config.yaml"

    train_mean_vector = np.load(train_mean_vector_path)
    config = load_config(config_path)

    yolo_processor = YoloBackboneAndPreprocess(config)
    ood_threshold = 240

    # Initialize evaluator
    evaluator = YOLOEvaluatorWithOOD(model_path, train_mean_vector, yolo_processor, ood_threshold=ood_threshold)

    # Input and output folders
    input_folder = '/home/carla/yongzhao/thesis/finalevaluation/7towns/noise_images_for_performance/town6_mosaic_num6_average/'
    output_folder = '/home/carla/yongzhao/thesis/finalevaluation/testnoise/'  # Replace with your desired output folder path

    # Run the filter and copy function
    filter_and_copy_images_with_ood_detection(evaluator, input_folder, output_folder)

    print("Filtering and copying of images completed.")
