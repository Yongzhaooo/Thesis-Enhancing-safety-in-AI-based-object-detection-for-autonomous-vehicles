'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\Unsupervised_together_with_yolo_backbone\\sort\\data_processing\\BiasCalculator.py
Author       : Yongzhao Chen
Date         : 2024-09-01 20:29:18
LastEditTime : 2024-09-01 22:38:06
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Description:
This script defines the `BiasCalculator` class, which is responsible for calculating bias values for images 
and folders based on feature vectors. It performs the following steps:
1. Loads the training feature center (`train_center.npy`) for bias calculation.
2. Calculates the bias for individual images and entire folders.
3. Computes histogram information for bias distribution.
4. Saves bias results and histogram data to JSON files.
5. Plots and saves histogram visualizations for bias distribution.

The script is designed for analyzing the bias distribution of images in noisy datasets.
'''

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import chain
import os

class BiasCalculator:
    def __init__(self, train_center_path, noise_folder, yolo_processor):
        """
        Initialize the BiasCalculator with the training center, noise folder, and YOLO processor.

        Args:
            train_center_path (str): Path to the training feature center file (`train_center.npy`).
            noise_folder (str): Path to the folder containing noisy images.
            yolo_processor (object): YOLO processor for calculating feature vectors.
        """
        # Load the training feature center
        self.train_center = np.load(train_center_path)
        self.noise_folder = noise_folder
        self.yolo_processor = yolo_processor
        
        # Initialize histinfo and record the name of the noise folder
        self.histinfo = {
            'noise_folder': Path(self.noise_folder).name  # Get the name of the noise folder
        }

    def calculate_bias_for_image(self, image_path):
        """
        Calculate the bias for a single image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            float: The calculated bias value.
        """
        feature_vector = self.yolo_processor.calculate_feature_vector(image_path).cpu().numpy().flatten()
        bias = np.linalg.norm(feature_vector - self.train_center)
        return bias

    def calculate_bias_for_folder(self, output_json="bias_results.json"):
        """
        Calculate the bias for all images in a folder and save the results.

        Args:
            output_json (str): Path to save the bias results JSON file.

        Returns:
            tuple: Average bias and a list of bias values for all images.
        """
        bias_results = {}
        image_folder = Path(self.noise_folder) / 'images'
        bias_list = []

        # Iterate through all images and calculate bias
        image_paths = chain(image_folder.glob('*.png'), image_folder.glob('*.jpg'))
        for image_path in image_paths:
            bias = self.calculate_bias_for_image(image_path)
            bias_results[str(image_path)] = float(bias)
            bias_list.append(bias)

        # Calculate the average bias
        avg_bias = float(np.mean(bias_list))

        # Save the statistics, including the average bias
        stats = {
            'bias_results': bias_results,
            'average_bias': avg_bias,
            'total_images': len(bias_list)
        }
        with open(output_json, 'w') as f:
            json.dump(stats, f, indent=4)

        return avg_bias, bias_list

    def compute_histinfo(self, bias_list, bin_size=2):
        """
        Compute the histogram information for the bias distribution.

        Args:
            bias_list (list): List of bias values.
            bin_size (int): Size of the bins for the histogram.

        Returns:
            dict: Histogram information including bin edges and counts.
        """
        # Compute histogram information
        hist, bin_edges = np.histogram(bias_list, bins=np.arange(min(bias_list), max(bias_list) + bin_size, bin_size))

        # Update histinfo with bin edges and counts
        self.histinfo['bin_edges'] = bin_edges.tolist()
        self.histinfo['hist_counts'] = hist.tolist()

        return self.histinfo

    def save_histinfo(self, output_json="histinfo.json"):
        """
        Save the histogram information to a JSON file.

        Args:
            output_json (str): Path to save the histogram JSON file.
        """
        # Save the histinfo file
        with open(output_json, 'w') as f:
            json.dump(self.histinfo, f, indent=4)
        print(f"Histogram information saved to {output_json}")

    def plot_histogram(self, x_min=120, x_max=600, y_max=50, save_filename="bias_histogram.png"):
        """
        Plot the histogram based on the histogram information.

        Args:
            x_min (int): Minimum value for the x-axis.
            x_max (int): Maximum value for the x-axis.
            y_max (int): Maximum value for the y-axis.
            save_filename (str): Filename to save the histogram plot.
        """
        bin_edges = self.histinfo.get('bin_edges', [])
        hist_counts = self.histinfo.get('hist_counts', [])

        if not bin_edges or not hist_counts:
            print("No histogram data available. Please compute histinfo first.")
            return

        plt.bar(bin_edges[:-1], hist_counts, width=np.diff(bin_edges), edgecolor='black', align='edge')
        
        if x_min is not None and x_max is not None:
            plt.xlim([x_min, x_max])
        if y_max is not None:
            plt.ylim([0, y_max])

        plt.title("Bias Distribution")
        plt.xlabel("Bias")
        plt.ylabel("Frequency")
        plt.grid(True)

        parent_folder = Path(self.noise_folder).parent
        plot_folder = parent_folder / "plots"
        plot_folder.mkdir(parents=True, exist_ok=True)

        save_path = plot_folder / save_filename
        plt.savefig(save_path)
        plt.show()
        print(f"Histogram saved to {save_path}")
