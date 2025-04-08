'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\Unsupervised_together_with_yolo_backbone\\sort\\drawtools\\HistSeperate.py
Author       : Yongzhao Chen
Date         : 2024-09-01 20:29:18
LastEditTime : 2024-09-01 22:38:06
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Description:
This script defines the `HistSeperate` class, which is responsible for loading and visualizing histogram data 
from `histinfo.json` files in a specified directory. It performs the following steps:
1. Recursively searches for `histinfo.json` files in the given directory.
2. Loads histogram data and categorizes it into two groups:
   - Raw data
   - Other data
3. Plots two separate histograms:
   - One for raw data
   - One for other data (dashed lines)

Note:
This script is not fully matured. It demonstrates another possible way of analyzing histogram data but may require further refinement for practical use.
'''

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

class HistSeperate:
    def __init__(self, top_level_layer):
        """Initialize the class with the top-level directory."""
        self.top_level_layer = top_level_layer
        self.histinfo_files = []

    def find_all_histinfo(self):
        """Recursively search for all `histinfo.json` file paths."""
        for root, dirs, files in os.walk(self.top_level_layer):
            histinfo_path = os.path.join(root, 'histinfo.json')
            if os.path.exists(histinfo_path):
                self.histinfo_files.append(histinfo_path)
        print(f"Found {len(self.histinfo_files)} histinfo files.")

    def plot_histograms(self):
        """Read all `histinfo.json` files and plot two separate histograms."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

        for histinfo_file in self.histinfo_files:
            # Read the `histinfo.json` file
            with open(histinfo_file, 'r') as f:
                histinfo = json.load(f)
            
            # Extract bin_edges and hist_counts
            bin_edges = np.array(histinfo['bin_edges'])
            hist_counts = np.array(histinfo['hist_counts'])

            # Extract folder information using the last two levels of the path
            folder_parts = Path(histinfo_file).parts
            folder_name = f"{folder_parts[-3]}/{folder_parts[-2]}"

            # Plot based on the file path
            if '/raw/histinfo' in histinfo_file:
                # If the file path contains `/raw/histinfo`, plot on the top figure with full opacity
                ax1.bar(bin_edges[:-1], hist_counts, width=np.diff(bin_edges), edgecolor='black', align='edge', alpha=1.0, label=folder_name)
            else:
                # For other cases, plot on the bottom figure with dashed lines
                ax2.step(bin_edges[:-1], hist_counts, where='mid', linestyle='--', label=f"{folder_parts[-2]}", alpha=0.7)

        # Configure the top figure (Raw Data Histogram)
        ax1.set_title("Raw Data Histogram")
        ax1.set_xlabel("Bias")
        ax1.set_ylabel("Frequency")
        ax1.legend(loc='upper right')
        ax1.grid(True)

        # Configure the bottom figure (Other Data Histogram)
        ax2.set_title("Other Data Histogram (Dashed Lines)")
        ax2.set_xlabel("Bias")
        ax2.set_ylabel("Frequency")
        ax2.legend(loc='upper right')
        ax2.grid(True)

        # Save the plot
        output_file = os.path.join(self.top_level_layer, 'combined_histogram.png')
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()
        print(f"Histogram saved to {output_file}")

# Example usage
if __name__ == "__main__":
    # Define the top-level directory
    top_level_layer = '/home/carla/yongzhao/thesis/finalevaluation/7towns/remix'
    
    # Create an instance of HistSeperate
    plotter = HistSeperate(top_level_layer)
    
    # Find all paths containing `histinfo.json`
    plotter.find_all_histinfo()
    
    # Plot all histograms
    plotter.plot_histograms()
