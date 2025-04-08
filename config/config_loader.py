'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\sort\\config\\config_loader.py
Author       : Yongzhao Chen
Date         : 2024-09-01 13:06:59
LastEditTime : 2024-09-09 15:59:27
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Description:
This script provides utility functions for loading global configuration files and noise configuration files.
It performs the following steps:
1. Loads the global configuration file (`config.yaml`) using the `load_config` function.
2. Processes the noise configuration sequence specified in the global configuration file.
3. Loads individual noise configuration files in JSON format using the `load_noise_config` function.
4. Combines all noise configurations into the global configuration dictionary for further use.

The script is designed to centralize configuration management for the project.
'''

import yaml
import json
import os

def load_config(config_path='config/config.yaml'):
    """
    Load the global configuration file.

    Args:
        config_path (str): Path to the global configuration file. Defaults to 'config/config.yaml'.

    Returns:
        dict: Parsed global configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Process the noise configuration sequence
    noise_configs = []
    for noise_config_name in config['noise_processing_sequence']:
        noise_config = load_noise_config(noise_config_name)
        noise_configs.append(noise_config)
    
    # Save all noise configurations in the global configuration
    config['noise_configs'] = noise_configs
    return config

def load_noise_config(config_name):
    """
    Load a noise configuration file based on the given name.

    Args:
        config_name (str): Name of the noise configuration file (without extension).

    Returns:
        dict: Parsed noise configuration dictionary.
    """
    noise_config_path = os.path.join('config/noise_configs', f'{config_name}.json')
    with open(noise_config_path, 'r') as file:
        noise_config = json.load(file)
    return noise_config

# Example usage
if __name__ == "__main__":
    # Load the global configuration file and print it
    config = load_config()
    print(config)
