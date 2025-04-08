'''
Copyright    : yongzhao.derek@gmail.com
FilePath     : \\sort\\cleansubfolders.py
Author       : Yongzhao Chen
Date         : 2024-09-09 12:20:35
LastEditTime : 2024-09-09 12:23:17
LastEditors  : Yongzhao Chen && yongzhao.derek@gmail.com
Version      : 1.0
Describe & Note: 
This script is used to clean up unnecessary files and folders within subdirectories of a specified top-level directory.
It performs the following steps:
1. Identifies all subfolders within the top-level directory.
2. Uses the `FolderCleaner` class to clean each subfolder by removing extra files and folders.
3. Outputs the progress for each processed folder.

The script is designed for batch cleaning of multiple subfolders in a directory.
'''

import os
from cleaner_tool.FolderCleaner import FolderCleaner

if __name__ == "__main__":
    # Define the top-level directory
    top_level_layer = '/home/carla/yongzhao/thesis/finalevaluation/7towns/30pics'
    config_path = 'config/config.yaml'

    # Initialize the FolderCleaner instance
    cleaner = FolderCleaner()

    # Get a list of all subfolders in the top-level directory
    folders_list = [f for f in os.listdir(top_level_layer) if os.path.isdir(os.path.join(top_level_layer, f))]

    for folder_name in folders_list:
        input_folder_path = os.path.join(top_level_layer, folder_name)
        print(f"Processing folder: {input_folder_path}")
        # Clean the current subfolder
        cleaner.clear_extra_folders(input_folder_path)

    print("All folders processed.")
