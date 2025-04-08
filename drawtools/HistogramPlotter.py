'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\Unsupervised_together_with_yolo_backbone\\sort\\drawtools\\HistogramPlotter.py
Author       : Yongzhao Chen
Date         : 2024-09-01 20:29:18
LastEditTime : 2024-09-01 22:38:06
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Description:
This script defines the `HistogramPlotter` class, which is responsible for loading, processing, and visualizing 
histogram data from various datasets. It performs the following steps:
1. Recursively searches for `histinfo.json` files in specified directories.
2. Loads histogram data for different categories:
   - Town10 raw data
   - Town10 noise data (Gaussian and Mosaic)
   - Non-Town10 raw data
   - Extra unrelated data (optional)
3. Aggregates histogram data by aligning it to global bin edges.
4. Plots aggregated histograms for comparison between different datasets.
5. Saves the resulting plots as PNG files.

The script is designed for analyzing and visualizing bias distributions across multiple datasets.
'''

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class HistogramPlotter:
    def __init__(self, top_level_layer, extra_data_base=False, extra_data_top_level_folder=None):
        self.top_level_layer = top_level_layer
        self.extra_data_base = extra_data_base
        self.extra_data_top_level_folder = extra_data_top_level_folder
        self.town10_raw_hist_data = []
        self.town10_noise_hist_data = []
        self.non_town10_raw_hist_data = []
        self.extra_hist_data = []  # Stores histogram data for extra unrelated datasets

    def find_histinfo(self, base_folder, filter_func=None):
        """Recursively search for `histinfo.json` files and store them in a list."""
        histinfo_files = []
        for root, dirs, files in os.walk(base_folder):
            histinfo_path = os.path.join(root, 'histinfo.json')
            if os.path.exists(histinfo_path):
                if filter_func is None or filter_func(root):
                    histinfo_files.append(histinfo_path)
        return histinfo_files

    def load_histogram_data(self):
        """Load and categorize histogram data from various datasets."""
        all_bin_edges = []

        # 1. Load Town10 raw data
        town10_raw_folder = os.path.join(self.top_level_layer, "Town10HD_Opt/with_noise/raw/")
        town10_raw_histinfo_files = self.find_histinfo(town10_raw_folder)
        for histinfo_file in town10_raw_histinfo_files:
            with open(histinfo_file, 'r') as f:
                histinfo = json.load(f)
            bin_edges = np.array(histinfo['bin_edges'])
            hist_counts = np.array(histinfo['hist_counts'])
            all_bin_edges.append(bin_edges)
            self.town10_raw_hist_data.append((bin_edges, hist_counts))

        # 2. Load Town10 noise data (Gaussian and Mosaic)
        town10_noise_folders = [
            os.path.join(self.top_level_layer, "Town10HD_Opt/with_noise/gaussian/"),
            os.path.join(self.top_level_layer, "Town10HD_Opt/with_noise/mosaic/")
        ]
        for folder in town10_noise_folders:
            noise_histinfo_files = self.find_histinfo(folder)
            for histinfo_file in noise_histinfo_files:
                with open(histinfo_file, 'r') as f:
                    histinfo = json.load(f)
                bin_edges = np.array(histinfo['bin_edges'])
                hist_counts = np.array(histinfo['hist_counts'])
                all_bin_edges.append(bin_edges)
                self.town10_noise_hist_data.append((bin_edges, hist_counts))

        # 3. Load Non-Town10 raw data
        non_town10_histinfo_files = self.find_histinfo(
            self.top_level_layer, 
            filter_func=lambda root: "Town10HD_Opt" not in root
        )
        for histinfo_file in non_town10_histinfo_files:
            with open(histinfo_file, 'r') as f:
                histinfo = json.load(f)
            bin_edges = np.array(histinfo['bin_edges'])
            hist_counts = np.array(histinfo['hist_counts'])
            all_bin_edges.append(bin_edges)
            self.non_town10_raw_hist_data.append((bin_edges, hist_counts))

        # 4. Load extra unrelated data
        if self.extra_data_base and self.extra_data_top_level_folder:
            extra_histinfo_files = self.find_histinfo(self.extra_data_top_level_folder)
            for histinfo_file in extra_histinfo_files:
                with open(histinfo_file, 'r') as f:
                    histinfo = json.load(f)
                bin_edges = np.array(histinfo['bin_edges'])
                hist_counts = np.array(histinfo['hist_counts'])
                all_bin_edges.append(bin_edges)
                self.extra_hist_data.append((bin_edges, hist_counts))

        # Determine global bin edges
        min_edge = min([np.min(b) for b in all_bin_edges])
        max_edge = max([np.max(b) for b in all_bin_edges])
        self.global_bin_edges = np.linspace(min_edge, max_edge, num=20)
        print(f"Global bin edges: {self.global_bin_edges}")

    def rebin_histogram(self, bin_edges, hist_counts, global_bin_edges):
        """Rebin histogram counts to align with global bin edges."""
        new_hist_counts, _ = np.histogram(np.repeat(bin_edges[:-1], hist_counts), bins=global_bin_edges)
        return new_hist_counts

    def aggregate_histogram(self, hist_data):
        """Aggregate histogram data by aligning it to global bin edges."""
        if not hist_data:
            return None, None
        aggregated_counts = np.zeros(len(self.global_bin_edges) - 1)
        for bin_edges, hist_counts in hist_data:
            rebinned_counts = self.rebin_histogram(bin_edges, hist_counts, self.global_bin_edges)
            aggregated_counts += rebinned_counts
        total_counts = np.sum(aggregated_counts)
        if total_counts > 0:
            aggregated_counts = aggregated_counts / total_counts
        return self.global_bin_edges, aggregated_counts

    def plot_aggregated_histograms(self):
        """Plot aggregated histograms with different colors for each dataset."""
        plt.figure(figsize=(10, 6))

        # Get aggregated histograms for each dataset
        town10_raw_bin_edges, town10_raw_aggregated_counts = self.aggregate_histogram(self.town10_raw_hist_data)
        town10_noise_bin_edges, town10_noise_aggregated_counts = self.aggregate_histogram(self.town10_noise_hist_data)
        non_town10_bin_edges, non_town10_aggregated_counts = self.aggregate_histogram(self.non_town10_raw_hist_data)
        extra_bin_edges, extra_aggregated_counts = self.aggregate_histogram(self.extra_hist_data) if self.extra_data_base else (None, None)

        # Plot histograms
        if town10_raw_bin_edges is not None:
            plt.bar(town10_raw_bin_edges[:-1], town10_raw_aggregated_counts, width=np.diff(town10_raw_bin_edges), edgecolor='black', align='edge', alpha=0.8, label="Town10 Raw Data", color='blue')

        if town10_noise_bin_edges is not None:
            plt.bar(town10_noise_bin_edges[:-1], town10_noise_aggregated_counts, width=np.diff(town10_noise_bin_edges), edgecolor='red', align='edge', linestyle='--', alpha=0.5, label="Town10 Noise Data", color='orange')

        if non_town10_bin_edges is not None:
            plt.bar(non_town10_bin_edges[:-1], non_town10_aggregated_counts, width=np.diff(non_town10_bin_edges), edgecolor='gray', align='edge', alpha=0.5, label="Non-Town10 Raw Data", color='green')

        if extra_bin_edges is not None:
            plt.bar(extra_bin_edges[:-1], extra_aggregated_counts, width=np.diff(extra_bin_edges), edgecolor='gray', align='edge', alpha=0.5, label="Unrelated Data", color='purple')

        # Adjust axes
        max_x = max(np.max(town10_raw_bin_edges) if town10_raw_bin_edges is not None else 0,
                    np.max(town10_noise_bin_edges) if town10_noise_bin_edges is not None else 0,
                    np.max(non_town10_bin_edges) if non_town10_bin_edges is not None else 0,
                    np.max(extra_bin_edges) if extra_bin_edges is not None else 0)
        max_y = 1  # Normalized maximum value

        plt.xlim(140, max_x)
        plt.ylim(0, max_y * 1.1)  # Add some space

        plt.title("Aggregated Histogram (Town10 Raw vs Noise vs Non-Town10 vs Unrelated Data)")
        plt.xlabel("Bias")
        plt.ylabel("Normalized Frequency")
        plt.legend(loc='upper right')
        plt.grid(True)

        # Save the plot
        output_file = os.path.join(self.top_level_layer, 'aggregated_histogram_custom.png')
        plt.savefig(output_file)
        plt.show()
        print(f"Histogram saved to {output_file}")

# Example usage
if __name__ == "__main__":
    top_level_layer = '/home/carla/yongzhao/thesis/finalevaluation/7towns/remix'
    extra_data_folder = '/home/carla/yongzhao/thesis/finalevaluation/new_coco_train_500/'  # Example unrelated data directory
    
    plotter = HistogramPlotter(top_level_layer, extra_data_base=True, extra_data_top_level_folder=extra_data_folder)
    
    # Load all histogram data
    plotter.load_histogram_data()
    
    # Plot aggregated histograms
    plotter.plot_aggregated_histograms()
