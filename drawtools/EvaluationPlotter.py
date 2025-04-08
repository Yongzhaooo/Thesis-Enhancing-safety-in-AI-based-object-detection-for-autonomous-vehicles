'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\Unsupervised_together_with_yolo_backbone\\sort\\drawtools\\EvaluationPlotter.py
Author       : Yongzhao Chen
Date         : 2024-09-01 20:29:18
LastEditTime : 2024-09-01 22:38:06
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Description:
This script defines the `EvaluationPlotter` class, which is responsible for visualizing the evaluation results 
of different noise types (Gaussian, Mosaic) and raw data. It performs the following steps:
1. Recursively searches for `overall_evaluation_results.csv` files in the specified directory.
2. Parses the noise parameters from the CSV files and categorizes the data into Gaussian, Mosaic, and raw data.
3. Optionally loads unrelated data from an external folder for comparison.
4. Generates three plots:
   - Gaussian Bias vs IoU
   - Mosaic Bias vs IoU
   - Combined plot of Gaussian and Mosaic Bias vs IoU
5. Saves the plots as PNG files in the specified directory.

The script is designed for analyzing and visualizing the relationship between bias and IoU across different noise types.
'''

import os
import pandas as pd
import matplotlib.pyplot as plt

class EvaluationPlotter:
    def __init__(self, top_level_layer, extra_data_folder=None):
        self.top_level_layer = top_level_layer
        self.csv_files = []
        self.extra_data_folder = extra_data_folder  # Path to the folder containing unrelated data
        self.extra_data = []  # Stores (bias, iou) information for unrelated data

    def find_all_csv(self):
        """Recursively search for `overall_evaluation_results.csv` files in the directory."""
        for root, dirs, files in os.walk(self.top_level_layer):
            if 'results' in dirs:
                csv_path = os.path.join(root, 'results', 'overall_evaluation_results.csv')
                if os.path.exists(csv_path):
                    self.csv_files.append(csv_path)
        print(f"Found {len(self.csv_files)} CSV files.")

    def parse_params(self, params):
        """Parse noise parameters and return the noise type and corresponding label."""
        if params.startswith("gaussian_params"):
            # Extract sigma value
            sigma_value = params.split("sigma=")[-1].split(",")[0].strip()
            label = f"G_{{{sigma_value}}}"
            return 'gaussian', label
        elif params.startswith("mosaic_params"):
            # Extract num_mosaics value
            num_value = params.split("num_mosaics=")[-1].split(",")[0].strip()
            label = f"M_{{{num_value}}}"
            return 'mosaic', label
        else:
            return 'raw', 'raw'

    def load_extra_data(self):
        """Load unrelated data from the external folder."""
        if self.extra_data_folder:
            extra_csv_path = os.path.join(self.extra_data_folder, 'results', 'overall_evaluation_results.csv')
            if os.path.exists(extra_csv_path):
                df = pd.read_csv(extra_csv_path)
                for _, row in df.iterrows():
                    bias = row['Bias']
                    iou = row['Mean IoU Across All Classes']
                    self.extra_data.append((bias, iou))
            print(f"Loaded extra data from {self.extra_data_folder}")

    def plot_results(self):
        """Generate three plots: Gaussian, Mosaic, and Combined, and add unrelated data as scatter points."""
        gaussian_data = {}
        mosaic_data = {}
        raw_data = {}

        # Iterate through all CSV files
        for csv_file in self.csv_files:
            df = pd.read_csv(csv_file)

            # Use the folder name as the Town label
            folder_name = os.path.basename(os.path.dirname(os.path.dirname(csv_file)))

            # Categorize data into Gaussian, Mosaic, and Raw
            for _, row in df.iterrows():
                bias = row['Bias']
                iou = row['Mean IoU Across All Classes']
                noise_type, label = self.parse_params(str(row['Noise_Type_Params']))

                if noise_type == 'gaussian':
                    if folder_name not in gaussian_data:
                        gaussian_data[folder_name] = []
                    gaussian_data[folder_name].append((bias, iou, label))
                elif noise_type == 'mosaic':
                    if folder_name not in mosaic_data:
                        mosaic_data[folder_name] = []
                    mosaic_data[folder_name].append((bias, iou, label))
                else:  # Raw data
                    if folder_name not in raw_data:
                        raw_data[folder_name] = []
                    raw_data[folder_name].append((bias, iou, label))

        # Load unrelated data
        self.load_extra_data()

        # Plot Gaussian data
        plt.figure(figsize=(10, 6))
        for folder_name, data in gaussian_data.items():
            raw_item = raw_data.get(folder_name, [(None, None, None)])
            combined_data = raw_item + data
            combined_data = [d for d in combined_data if d[0] is not None]  # Exclude None
            combined_data.sort(key=lambda x: x[0])  # Sort by Bias

            biases, ious, labels = zip(*combined_data)
            plt.plot(biases, ious, marker='o', label=f"{folder_name}_G")

            # Add labels to each point
            for bias, iou, label in combined_data:
                if label == 'raw':
                    plt.text(bias, iou, f"raw", fontsize=5)
                else:
                    plt.text(bias, iou, f"{label}", fontsize=5)

        # Add unrelated data as scatter points
        if self.extra_data:
            extra_biases, extra_ious = zip(*self.extra_data)
            plt.scatter(extra_biases, extra_ious, color='gray', label="Unrelated Data", alpha=0.6)

        plt.title("Gaussian Bias vs IoU")
        plt.xlabel("Bias")
        plt.ylabel("Mean IoU Across All Classes")
        plt.legend(loc='best')
        plt.grid(True)
        gaussian_fig_path = os.path.join(self.top_level_layer, 'gaussian_plot.png')
        plt.savefig(gaussian_fig_path)
        print("Gaussian plot saved to:", gaussian_fig_path)
        plt.show()

        # Plot Mosaic data
        plt.figure(figsize=(10, 6))
        for folder_name, data in mosaic_data.items():
            raw_item = raw_data.get(folder_name, [(None, None, None)])
            combined_data = raw_item + data
            combined_data = [d for d in combined_data if d[0] is not None]  # Exclude None
            combined_data.sort(key=lambda x: x[0])  # Sort by Bias

            biases, ious, labels = zip(*combined_data)
            plt.plot(biases, ious, marker='o', label=f"{folder_name}_M")

            # Add labels to each point
            for bias, iou, label in combined_data:
                if label == 'raw':
                    plt.text(bias, iou, f"raw", fontsize=5)
                else:
                    plt.text(bias, iou, f"{label}", fontsize=5)

        # Add unrelated data as scatter points
        if self.extra_data:
            extra_biases, extra_ious = zip(*self.extra_data)
            plt.scatter(extra_biases, extra_ious, color='gray', label="Unrelated Data", alpha=0.6)

        plt.title("Mosaic Bias vs IoU")
        plt.xlabel("Bias")
        plt.ylabel("Mean IoU Across All Classes")
        plt.legend(loc='best')
        plt.grid(True)
        mosaic_fig_path = os.path.join(self.top_level_layer, 'mosaic_plot.png')
        plt.savefig(mosaic_fig_path)
        print("Mosaic plot saved to:", mosaic_fig_path)
        plt.show()

        # Plot Combined data
        plt.figure(figsize=(10, 6))
        # Gaussian
        for folder_name, data in gaussian_data.items():
            raw_item = raw_data.get(folder_name, [(None, None, None)])
            combined_data = raw_item + data
            combined_data = [d for d in combined_data if d[0] is not None]  # Exclude None
            combined_data.sort(key=lambda x: x[0])

            biases, ious, labels = zip(*combined_data)
            plt.plot(biases, ious, marker='o', label=f"Gaussian - {folder_name}")

            # Add labels to each point
            for bias, iou, label in combined_data:
                if label == 'raw':
                    plt.text(bias, iou, "raw", fontsize=5)
                else:
                    plt.text(bias, iou, f"{label}", fontsize=5)

        # Mosaic
        for folder_name, data in mosaic_data.items():
            raw_item = raw_data.get(folder_name, [(None, None, None)])
            combined_data = raw_item + data
            combined_data = [d for d in combined_data if d[0] is not None]  # Exclude None
            combined_data.sort(key=lambda x: x[0])

            biases, ious, labels = zip(*combined_data)
            plt.plot(biases, ious, marker='o', label=f"Mosaic - {folder_name}")

            # Add labels to each point
            for bias, iou, label in combined_data:
                if label == 'raw':
                    plt.text(bias, iou, "raw", fontsize=5)
                else:
                    plt.text(bias, iou, f"{label}", fontsize=5)

        # Add unrelated data as scatter points
        if self.extra_data:
            extra_biases, extra_ious = zip(*self.extra_data)
            plt.scatter(extra_biases, extra_ious, color='gray', label="Unrelated Data", alpha=0.6)

        plt.title("Gaussian and Mosaic Bias vs IoU")
        plt.xlabel("Bias")
        plt.ylabel("Mean IoU Across All Classes")
        plt.legend(loc='best')
        plt.grid(True)
        combined_fig_path = os.path.join(self.top_level_layer, 'combined_plot.png')
        plt.savefig(combined_fig_path)
        print("Combined plot saved to:", combined_fig_path)
        plt.show()

# Example usage
if __name__ == "__main__":
    top_level_layer = '/home/carla/yongzhao/thesis/finalevaluation/7towns/remix'
    extra_data_folder = '/home/carla/yongzhao/thesis/finalevaluation/new_coco_train_500/'  
    
    plotter = EvaluationPlotter(top_level_layer, extra_data_folder=extra_data_folder)
    
    # Find all paths containing `overall_evaluation_results.csv`
    plotter.find_all_csv()
    
    # Generate three plots: Gaussian, Mosaic, and Combined
    plotter.plot_results()
    print("All plots saved.")
