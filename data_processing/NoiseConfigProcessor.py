'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\Unsupervised_together_with_yolo_backbone\\sort\\data_processing\\NoiseConfigProcessor.py
Author       : Yongzhao Chen
Date         : 2024-09-01 20:29:18
LastEditTime : 2024-09-01 22:38:06
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Description:
This script defines the `NoiseConfigProcessor` class, which is responsible for managing and processing noise 
configuration files. It performs the following steps:
1. Loads global configuration settings from a YAML file.
2. Ensures the existence of the `noise_configs` directory.
3. Loads individual noise configuration files in JSON format.
4. Generates noise levels for Gaussian and Mosaic noise based on configuration parameters.
5. Merges multiple noise configuration files into a single JSON file for further processing.

The script is designed for managing noise configurations and generating noise parameters for dataset augmentation.
'''

import os
import json
import yaml

class NoiseConfigProcessor:
    def __init__(self, config_path='config/config.yaml'):
        """
        Initialize and load the global configuration file.

        Args:
            config_path (str): Path to the global configuration file. Defaults to 'config/config.yaml'.
        """
        self.config = self.load_config(config_path)
        self.noise_configs_dir = os.path.join(os.path.dirname(config_path), 'noise_configs')
        
        # Ensure the `noise_configs` directory exists
        if not os.path.exists(self.noise_configs_dir):
            raise FileNotFoundError(f"Noise configuration directory does not exist: {self.noise_configs_dir}")
        
    def load_config(self, config_path):
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

    def ensure_directory_exists(self, directory):
        """
        Ensure the specified directory exists. If not, create it.

        Args:
            directory (str): Path to the directory.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

    def generate_noise_levels(self, initial_value, delta, num_levels):
        """
        Generate a list of noise levels.

        Args:
            initial_value (float): Initial value for the noise level.
            delta (float): Increment for each noise level.
            num_levels (int): Number of noise levels to generate.

        Returns:
            list: List of generated noise levels.
        """
        return [initial_value + i * delta for i in range(num_levels)]

    def load_noise_config(self, config_name):
        """
        Load a single noise configuration file.

        Args:
            config_name (str): Name of the noise configuration file (without extension).

        Returns:
            dict: Parsed noise configuration dictionary.
        """
        config_file_path = os.path.join(self.noise_configs_dir, f'{config_name}.json')
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Noise configuration file does not exist: {config_file_path}")
        
        with open(config_file_path, 'r') as file:
            noise_config = json.load(file)
        return noise_config

    def merge_noise_configs(self, output_path='config/noise_configs/noise_params.json'):
        """
        Merge multiple noise configuration files and generate incremental noise sequences. Save the result as a JSON file.

        Args:
            output_path (str): Path to save the merged noise configuration file.
        """
        noise_sequence = self.config['noise_processing_sequence']
        
        # Container for all noise parameters
        merged_noise_params = {
            "gaussian": [],
            "mosaic": []
        }

        # Iterate through and merge each noise configuration file
        for noise_config_name in noise_sequence:
            noise_config = self.load_noise_config(noise_config_name)
            
            # Generate Gaussian noise levels, keeping the mean constant
            gaussian_levels = self.generate_noise_levels(
                noise_config['Initial_Gaussian_SIGMA'], 
                noise_config['DELTA_LEVELS_Gaussian'], 
                noise_config['NUM_LEVELS_Gaussian']
            )
            
            # Add Gaussian noise parameters to the result
            merged_noise_params['gaussian'].extend([{
                "mean": noise_config['Gaussian_MEAN'],
                "sigma": sigma
            } for sigma in gaussian_levels])

            # Generate Mosaic noise levels
            mosaic_levels = self.generate_noise_levels(
                noise_config['Initial_Mosaic_NUM'], 
                noise_config['DELTA_LEVELS_MOSAIC'], 
                noise_config['NUM_LEVELS_MOSAIC']
            )
            
            # Add Mosaic noise parameters to the result
            merged_noise_params['mosaic'].extend([{
                "num_mosaics": num_mosaics,
                "min_mosaic_size": noise_config['MIN_MOSAIC_SIZE'],
                "max_mosaic_size": noise_config['MAX_MOSAIC_SIZE'],
                "color_mode": noise_config['COLOR_MODE']
            } for num_mosaics in mosaic_levels])

        # Save the merged noise parameters as a JSON file
        self.ensure_directory_exists(os.path.dirname(output_path))
        with open(output_path, 'w') as f:
            json.dump(merged_noise_params, f, indent=4)
        print(f"Merged noise parameters saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Initialize the processor and load the configuration file
    processor = NoiseConfigProcessor()

    # Merge noise configurations and save as a JSON file
    processor.merge_noise_configs()
