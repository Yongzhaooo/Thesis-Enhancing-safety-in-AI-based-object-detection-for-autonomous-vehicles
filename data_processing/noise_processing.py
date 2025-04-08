'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\Unsupervised_together_with_yolo_backbone\\sort\\data_processing\\noise_processing.py
Author       : Yongzhao Chen
Date         : 2024-09-01 20:29:18
LastEditTime : 2024-09-01 22:38:06
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Description:
This script defines the `ImageNoiseAdder` class, which is responsible for adding noise (Gaussian or Mosaic) 
to images in a dataset. It performs the following steps:
1. Loads configuration settings from a YAML file.
2. Ensures the existence of output directories for processed images and labels.
3. Applies Gaussian or Mosaic noise to images in batches.
4. Saves the processed images to the corresponding output directories.
5. Copies the label files to the output directories.

The script is designed for generating noisy datasets for evaluation and training purposes.
'''

import os
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
import random
from pathlib import Path
import yaml

def load_config(config_path='config/config.yaml'):
    """Load the global configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def ensure_directory_exists(directory):
    """Ensure the specified directory exists. If not, create it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Ensured directory exists: {directory}")
        
class ImageNoiseAdder:
    def __init__(self, input_folder_path, output_folder_path, config):
        """
        Initialize the ImageNoiseAdder with input and output folder paths and configuration.

        Args:
            input_folder_path (str): Path to the input folder containing images and labels.
            output_folder_path (str): Path to the output folder for saving processed images and labels.
            config (dict): Configuration settings loaded from a YAML file.
        """
        self.config = config
        self.input_folder = Path(input_folder_path)
        self.output_base_folder = Path(output_folder_path)
        self.raw_images_folder = self.input_folder / "images"
        self.output_folder = None

    @staticmethod
    def add_gaussian_noise(image, mean, sigma):
        """
        Add Gaussian noise to an image.

        Args:
            image (numpy.ndarray): Input image.
            mean (float): Mean of the Gaussian noise.
            sigma (float): Standard deviation of the Gaussian noise.

        Returns:
            numpy.ndarray: Image with Gaussian noise added.
        """
        row, col, ch = image.shape
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy_image = image + gauss
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image

    @staticmethod
    def add_random_mosaics(image, num_mosaics, min_mosaic_size, max_mosaic_size, color_mode="average"):
        """
        Add random mosaic blocks to an image.

        Args:
            image (numpy.ndarray): Input image.
            num_mosaics (int): Number of mosaic blocks to add.
            min_mosaic_size (int): Minimum size of a mosaic block.
            max_mosaic_size (int): Maximum size of a mosaic block.
            color_mode (str): Color mode for the mosaic blocks ("average" or "black").

        Returns:
            numpy.ndarray: Image with mosaic blocks added.
        """
        rows, cols, _ = image.shape
        for _ in range(num_mosaics):
            mosaic_size = random.randint(min_mosaic_size, max_mosaic_size)
            x1 = random.randint(0, cols - mosaic_size)
            y1 = random.randint(0, rows - mosaic_size)
            x2 = x1 + mosaic_size
            y2 = y1 + mosaic_size

            block = image[y1:y2, x1:x2]
            if color_mode == "average":
                color = block.mean(axis=(0, 1)).astype(int)
            elif color_mode == "black":
                color = [0, 0, 0]
            else:
                raise ValueError(f"Unsupported color mode: {color_mode}")
            image[y1:y2, x1:x2] = color
        return image

    def add_noise_and_save_image(self, image, filename, noise_type, params):
        """
        Process and save a single image with added noise.

        Args:
            image (numpy.ndarray): Input image.
            filename (str): Name of the image file.
            noise_type (str): Type of noise to add ("gaussian" or "mosaic").
            params (dict): Parameters for the noise type.
        """
        if noise_type == "gaussian":
            noisy_image = self.add_gaussian_noise(image, **params)
        elif noise_type == "mosaic":
            noisy_image = self.add_random_mosaics(image, **params)
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        output_image_path = self.output_folder / 'images' / filename
        cv2.imwrite(str(output_image_path), noisy_image)

    def initialize_output_folder_for_noise_type(self, noise_type, **params):
        """
        Set up the output folder based on the noise type and parameters.

        Args:
            noise_type (str): Type of noise ("gaussian" or "mosaic").
            params (dict): Parameters for the noise type.
        """
        if noise_type == "gaussian":
            folder_name = f"mean{params['mean']}_sigma{params['sigma']}"
        elif noise_type == "mosaic":
            folder_name = f"num{params['num_mosaics']}_{params['color_mode']}"
        else:
            raise ValueError(f"Unsupported noise type: {noise_type}")

        self.output_folder = self.output_base_folder / noise_type / folder_name
        self.output_folder.mkdir(parents=True, exist_ok=True)
        (self.output_folder / "images").mkdir(exist_ok=True)
        (self.output_folder / "labels").mkdir(exist_ok=True)

    def apply_noise_to_images(self, noise_type="gaussian", batch_size=None, **params):
        """
        Add noise to all images and save them to the output folder.

        Args:
            noise_type (str): Type of noise to add ("gaussian" or "mosaic").
            batch_size (int): Number of images to process in a batch.
            params (dict): Parameters for the noise type.
        """
        if batch_size is None:
            batch_size = self.config['noise_batch']

        self.initialize_output_folder_for_noise_type(noise_type, **params)
        print(f"Applying {noise_type} noise with parameters: {params}")

        images_batch = []
        filenames_batch = []

        pool = Pool(processes=cpu_count() - 1)
        results = []

        try:
            for filename in tqdm(os.listdir(self.raw_images_folder), desc=f"Processing {noise_type} noise"):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = self.raw_images_folder / filename
                    image = cv2.imread(str(image_path))
                    if image is None:
                        print(f"Failed to read {filename}. Skipping...")
                        continue

                    images_batch.append(image)
                    filenames_batch.append(filename)

                    if len(images_batch) >= batch_size:
                        for img, fname in zip(images_batch, filenames_batch):
                            result = pool.apply_async(self.add_noise_and_save_image, (img, fname, noise_type, params))
                            results.append(result)
                        images_batch.clear()
                        filenames_batch.clear()

            # Process remaining images
            if images_batch:
                for img, fname in zip(images_batch, filenames_batch):
                    result = pool.apply_async(self.add_noise_and_save_image, (img, fname, noise_type, params))
                    results.append(result)
        finally:
            pool.close()
            pool.join()
            for result in results:
                result.get()  # Ensure all async tasks are completed

            self.copy_image_labels()

    def copy_image_labels(self):
        """
        Copy label files to the corresponding noise folder.
        """
        src_label_folder = self.input_folder / "labels"
        dest_label_folder = self.output_folder / "labels"

        if not dest_label_folder.exists():
            dest_label_folder.mkdir(parents=True, exist_ok=True)

        for label_file in os.listdir(src_label_folder):
            src_file_path = src_label_folder / label_file
            dest_file_path = dest_label_folder / label_file
            shutil.copy2(src_file_path, dest_file_path)
